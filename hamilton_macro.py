import os

import pandas as pd
import runhouse as rh

from embedder import URLEmbedder
from scrape import extract_urls


def urls(base_url: str, max_scrape_depth: int = 1, cutoff: int = None) -> pd.DataFrame:
    """Gives all recursive URLs from the given base URL."""
    all_urls = extract_urls(base_url, max_depth=max_scrape_depth)
    relevant_urls = all_urls[:cutoff] if cutoff else all_urls
    return pd.DataFrame(relevant_urls, columns=["url"])


def cluster() -> rh.Cluster:
    """Sets up a cluster for embedding URLs."""
    return rh.cluster(f"/dongreenberg/rh-hamilton-a10g",
                      instance_type="A10G:1",
                      auto_stop_mins=5,
                      spot=True).up_if_not()


def embedder(cluster: rh.Cluster) -> URLEmbedder:
    """Sets up an embedder to embed URLs on a remote GPU box."""
    env = rh.env(
        name=f"langchain_embed_env",
        reqs=["langchain", "langchain-community", "langchainhub", "sentence_transformers", "bs4"],
    )
    RemoteURLEmbedder = rh.module(URLEmbedder).to(cluster, env)
    return RemoteURLEmbedder(
        model_name_or_path="BAAI/bge-large-en-v1.5",
        device="cuda",
        name=f"doc_embedder",
    )


def embeddings_df(urls: pd.DataFrame, embedder: URLEmbedder) -> pd.DataFrame:
    """Adds embeddings to the given URL DataFrame."""
    urls["embeddings"] = [embedder.embed(url, normalize_embeddings=True) for url in urls["url"]]
    return urls


def saved_embeddings(embeddings_df: pd.DataFrame) -> str:
    """Saves the embeddings to disk."""
    branch = os.environ.get("GITHUB_HEAD_REF", "main")
    os.makedirs(branch, exist_ok=True)
    path = f"{branch}/url_embeddings.csv"
    embeddings_df.to_csv(path, index=False)

    if branch != "main":
        rh.folder(path=f"{branch}").to(system="s3", path=f"runhouse/url_embeddings/{branch}")
        path = f"s3://runhouse/url_embeddings/{branch}/url_embeddings.csv"
    return path