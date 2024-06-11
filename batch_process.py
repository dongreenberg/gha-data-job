import os
import sys

import hamilton_sdk.adapters
import pandas as pd
import runhouse as rh
from hamilton import driver

from scrape import extract_urls
from embedder import URLEmbedder


def urls(base_url: str, max_scrape_depth: int = 1, cutoff: int = None) -> pd.DataFrame:
    """Gives all recursive URLs from the given base URL."""
    all_urls = extract_urls(base_url, max_depth=max_scrape_depth)
    relevant_urls = all_urls[:cutoff] if cutoff else all_urls
    return pd.DataFrame(relevant_urls, columns=["url"])


def embedder() -> URLEmbedder:
    """Sets up an embedder to embed URLs on a remote GPU box."""
    cluster = rh.cluster(f"/dongreenberg/rh-hamilton-a10g",
                      instance_type="A10G:1",
                      auto_stop_mins=5,
                      spot=True).up_if_not()
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


def _generate_url_embeddings(base_url):
    """This leverages Hamilton as a few big functions. The piece that runs the
    embeddings in parallel is all done inside a single function, so those assets are not visible.
    This begins with an underscore as hamilton, by default, crawls all fns in a module, ignoring _ prefixes"""
    import __main__
    dr = (
        driver
        .Builder()
        .with_adapters(
            hamilton_sdk.adapters.HamiltonTracker(
                project_id=1,
                username="elijah",
                dag_name="embeddings_workflow",
                # You can also connect to the hosted instance
                # api_key=os.environ.get("DAGWORKS_API_KEY"),
                # hamilton_api_url="https://api.app.dagworks.io",
                # hamilton_ui_url="https://app.dagworks.io",
            )
        )
        .with_modules(sys.modules[__name__])
        .build()
    )
    dr.visualize_execution(
        ["saved_embeddings"],
        bypass_validation=True,
        output_file_path="./hamilton_macro.png",
    )
    print(dr.execute(
        ["saved_embeddings"],
        inputs={
            "base_url": base_url,
            "max_scrape_depth": 1,
            "cutoff": 4
        }
    ))


if __name__ == "__main__":
    _generate_url_embeddings("https://en.wikipedia.org/wiki/Poker")
