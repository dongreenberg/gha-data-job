import os

import pandas as pd
import runhouse as rh

from scrape import extract_urls
from embedder import URLEmbedder


def generate_url_embeddings(base_url: str):
    # We recursively extract all children URLs from the given base URL.
    urls = extract_urls(base_url, max_depth=1)
    url_df = pd.DataFrame(urls, columns=["url"])

    # Set up an embedder to embed the URLs on the fly on a remote GPU box
    cluster = rh.cluster(f"rh-hamilton-a10g",
                         instance_type="A10G:1",
                         auto_stop_mins=5,
                         spot=True).up_if_not()
    env = rh.env(
        name=f"langchain_embed_env",
        reqs=["langchain", "langchain-community", "langchainhub", "sentence_transformers", "bs4"],
    )
    RemoteURLEmbedder = rh.module(URLEmbedder).to(cluster, env)
    embedder = RemoteURLEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
        name=f"doc_embedder",
    )

    # Add new column with the embeddings
    url_df["embeddings"] = [embedder.embed(url, normalize_embeddings=True) for url in urls]

    # Save the embeddings to disk
    url_df.to_csv("url_embeddings.csv", index=False)

    # Send to S3
    # Get the current git revision to track the code version
    branch = os.environ.get("GITHUB_HEAD_REF", "main")
    rh.folder(path="url_embeddings.csv").to(f"s3://url_embeddings/url_embeddings.csv@{branch}")


if __name__ == "__main__":
    generate_url_embeddings("https://en.wikipedia.org/wiki/Poker")
