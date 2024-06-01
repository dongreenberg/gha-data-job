import functools
import os

import hamilton_sdk.adapters
import pandas as pd
import runhouse as rh
from hamilton import driver
from hamilton.execution.executors import SynchronousLocalTaskExecutor

import h_runhouse
from scrape import extract_urls
from embedder import URLEmbedder


def generate_url_embeddings(base_url: str):
    # We recursively extract all children URLs from the given base URL.
    urls = extract_urls(base_url, max_depth=1)[:4]
    url_df = pd.DataFrame(urls, columns=["url"])

    # Set up an embedder to embed the URLs on the fly on a remote GPU box
    cluster = rh.cluster(f"/dongreenberg/rh-hamilton-a10g",
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
    # Get the current git revision to track the code version. If not running from GitHub, default to "main"
    branch = os.environ.get("GITHUB_HEAD_REF", "main")
    os.makedirs(branch, exist_ok=True)
    url_df.to_csv(f"{branch}/url_embeddings.csv", index=False)

    # If we're running from Github, save to S3
    if branch != "main":
        rh.folder(path=f"{branch}").to(system="s3", path=f"runhouse/url_embeddings/{branch}")


def generate_url_embeddings_hamilton_macro(base_url):
    """This leverages Hamilton as a few big functions. The piece that runs the
    embeddings in parallel is all done inside a single function, so those assets are not visible."""
    import hamilton_macro
    dr = (
        driver
        .Builder()
        .with_modules(hamilton_macro)
        .with_adapters(
            hamilton_sdk.adapters.HamiltonTracker(
                project_id=19374,
                username="elijah@dagworks.io",
                dag_name="runhouse_macro_version",
                api_key=os.environ.get("DAGWORKS_API_KEY"),
                hamilton_api_url="https://api.app.dagworks.io",
                hamilton_ui_url="https://app.dagworks.io",
            )
        )
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
            "base_url": "https://en.wikipedia.org/wiki/Poker",
            "max_scrape_depth": 1,
            "cutoff": 4
        }
    ))


def generate_url_embeddings_hamilton_micro(base_url):
    """This leverages Hamilton's parallelism capability + a runhouse delegating orchestrator
    to use Hamilton to handle parallelism + asset tracking, and runhouse to handle provisioning
    remote infra. This allows us fine-grained visibility into the pipeline's performance.

    """
    import hamilton_micro

    cluster = rh.cluster(f"/dongreenberg/rh-hamilton-a10g",
                         instance_type="A10G:1",
                         auto_stop_mins=5,
                         spot=True).up_if_not()
    env = rh.env(
        name=f"langchain_embed_env",
        reqs=[
            "langchain",
            "langchain-community",
            "langchainhub",
            "sentence_transformers",
            "bs4",
            "sf-hamilton"
        ],
    )
    model_params = {
        "model_name_or_path": "BAAI/bge-large-en-v1.5",
        "device": "cuda",
        # uncomment for local testing
        # "device": "cpu",
    }

    dr = (
        driver
        .Builder()
        .with_modules(hamilton_micro)
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_remote_executor(
            # uncomment for local testing
            # SynchronousLocalTaskExecutor()
            h_runhouse.RunhouseExecutor(
                max_tasks=4,
                cluster=cluster,
                env=env,
            )
        )
        .with_adapters(
            hamilton_sdk.adapters.HamiltonTracker(
                project_id=19374,
                username="elijah@dagworks.io",
                dag_name="runhouse_micro_version",
                api_key=os.environ.get("DAGWORKS_API_KEY"),
                hamilton_api_url="https://api.app.dagworks.io",
                hamilton_ui_url="https://app.dagworks.io",
            )
        )
        .build()
    )
    dr.visualize_execution(
        ["saved_embeddings"],
        bypass_validation=True,
        output_file_path="./hamilton_micro.png",
    )
    print(dr.execute(
        ["saved_embeddings"],
        inputs={
            "base_url": "https://en.wikipedia.org/wiki/Poker",
            "max_scrape_depth": 1,
            "cutoff": 20,
            # tuple for serialization so we can cache it
            "embed_params": dict(
                normalize_embeddings=True
            ),
            "model_params": model_params
        }
    ))


if __name__ == "__main__":
    # generate_url_embeddings("https://en.wikipedia.org/wiki/Poker")
    # generate_url_embeddings_hamilton_macro("https://en.wikipedia.org/wiki/Poker")
    generate_url_embeddings_hamilton_micro("https://en.wikipedia.org/wiki/Poker")
