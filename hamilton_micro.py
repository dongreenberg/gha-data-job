import functools
import os
from typing import Any

import pandas as pd
import runhouse as rh
from hamilton.htypes import Parallelizable, Collect
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from scrape import extract_urls

MODEL = None


def all_urls(base_url: str, max_scrape_depth: int = 1, cutoff: int = None) -> list[str]:
    """Gives all recursive URLs from the given base URL."""
    all_urls = extract_urls(base_url, max_depth=max_scrape_depth)
    relevant_urls = all_urls[:cutoff] if cutoff else all_urls
    return relevant_urls


def urls(all_urls: list[str]) -> Parallelizable[str]:
    """This is not strictly needed, but it sets up each task"""
    for url in all_urls:
        yield url


# Everything from here -> the next comment is a "subtask" that can be run in parallel

# underscore is a convention for non-crawled functions
# we put this behind an lru cache to handle state
@functools.lru_cache
def _initialize_model(**model_params) -> SentenceTransformer:
    return SentenceTransformer(**model_params)


def splits(urls: str) -> list[str]:
    """Split the documents into chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    docs = WebBaseLoader(
        web_paths=[urls],
    ).load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    splits_as_str = [doc.page_content for doc in splits]
    return splits_as_str


EmbeddingsWithText = tuple[list[str], list[list[float]]]


def embeddings(splits: list[str], embed_params: dict, model_params: dict[str, Any]) -> EmbeddingsWithText:
    """Embed the splits -- will instantiate a model if it hasn't been"""
    model = _initialize_model(**model_params)  # cached if needed
    return splits, model.encode(splits, **embed_params)


# This collects the results of the embeddings, moving it out of the parallel block
def embeddings_df(embeddings: Collect[EmbeddingsWithText], all_urls: list[str]) -> pd.DataFrame:
    """Adds embeddings to the given URL DataFrame."""
    embeddings_list = []
    for url, (splits, embedding) in zip(all_urls, embeddings):
        embeddings_list.append([url, embedding])
    urls_df = pd.DataFrame(embeddings_list, columns=["url", "embeddings"])
    return urls_df


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
