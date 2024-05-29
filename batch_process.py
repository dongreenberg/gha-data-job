import asyncio
import time
from typing import List

import runhouse as rh
from tqdm.asyncio import tqdm
import pandas as pd

from scrape import extract_urls


class URLEmbedder:
    def __init__(self, **model_kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(**model_kwargs)

    def embed_docs(self, urls: List[str], **embed_kwargs):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        start = time.time()
        docs = WebBaseLoader(
            web_paths=urls,
        ).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in splits]
        downloaded = time.time()
        embedding = self.model.encode(splits_as_str, **embed_kwargs)
        return urls[0], embedding, downloaded - start, time.time() - downloaded


# ## Setting up Runhouse primitives
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster.
async def main():

    # We set up some parameters for our embedding task.
    num_replicas = 4  # Number of models to load side by side
    max_concurrency_per_replica = 32  # Number of parallel calls to make to each replica
    url_to_recursively_embed = "https://en.wikipedia.org/wiki/Poker"

    # We recursively extract all children URLs from the given URL.
    start_time = time.time()
    urls = extract_urls(url_to_recursively_embed, max_depth=2)
    url_df = pd.DataFrame(urls, columns=["url"])
    print(f"Time to extract {len(urls)} URLs: {time.time() - start_time}")

    # First, we create a cluster with the desired instance type and provider.
    # Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count we need We could
    # alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`. However, we
    # provision `num_replicas` number of these instances. This gives us one Runhouse cluster that has
    # several separate GPU machines that it can access.
    #
    # This is one major advantage of Runhouse: you can use a multinode machine as if it were one opaque cluster,
    # and send things to it from your local machine. This is especially useful for embarrassingly parallel tasks
    # like this one. Note that it is also far easier to provision several A10G:1 machines as spot instances
    # than it is to provision a single A10G:4 machine, which is why we do it this way.
    #
    # Note that if the cluster was
    # already up (e.g. if we had run this script before), the code would just bring it up instead of creating a
    # new one, since we have given it a unique name `"rh-4xa10g"`.
    #
    # Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
    start_time = time.time()
    embedder_replicas = []
    cluster = rh.cluster(
        f"rh-{num_replicas}xa10g",
        instance_type="A10G:1",
        num_instances=num_replicas,
        spot=True,
    ).up_if_not()

    # Generally, when using Runhouse, you would initialize an env with `rh.env`, and send your module to
    # that env. Each env runs in a *separate process* on the cluster. In this case, we want to have 4 copies of the
    # embedding model in separate processes, because we have 4 GPUs. We can do this by creating 4 separate envs
    # and 4 separate modules, each sent to a separate env. We do this in a loop here, with a list of dependencies
    # that we need remotely to run the module.
    #
    # In this case, each `env` is also on a separate machine, but we could also provision an A10G:4 instance,
    # and send all 4 envs to the same machine. Each env runs within a separate process on the machine, so they
    # won't interfere with each other.
    #
    # Note that we send the `URLEmbedder` class to the cluster, and then can construct our modules using the
    # returned "remote class" instead of the normal local class. These instances are then actually constructed
    # on the cluster, and any methods called on these instances would run on the cluster.
    for i in range(num_replicas):
        env = rh.env(
            name=f"langchain_embed_env_{i}",
            reqs=[
                "langchain",
                "langchain-community",
                "langchainhub",
                "bs4",
                "sentence_transformers",
                "fake_useragent",
            ],
            compute={"GPU": 1},
        )
        RemoteURLEmbedder = rh.module(URLEmbedder).get_or_to(cluster, env)
        remote_url_embedder = RemoteURLEmbedder(
            model_name_or_path="BAAI/bge-large-en-v1.5",
            device="cuda",
            name=f"doc_embedder_{i}",
        )
        embedder_replicas.append(remote_url_embedder)
    print(f"Time to initialize {num_replicas} replicas: {time.time() - start_time}")

    # ## Calling the Runhouse modules in parallel
    # We'll simply use the `embed_docs` function on the remote module to embed all the URLs in parallel. Note that
    # we can call this function exactly as if it were a local module. The semaphore and asyncio logic allows us
    # to run all the functions in parallel, up to a maximum total concurrency.

    # We pass a few special arguments to the Runhouse function.
    #
    # We need to use a special `run_async=True`
    # argument to the function. This tells Runhouse to return a coroutine that we can await on, rather than making
    # a blocking network call to the server. This allows us to use `asyncio` logic locally to run all the functions
    # in parallel.
    #
    # We also pass `stream_logs=False`, which means we won't get the stdout/stderr of the remote
    # function on our local machine. In this case, we're running a large batch job, and don't want to slow down
    # our work by spamming our local machine with logs.
    semaphore = asyncio.Semaphore(max_concurrency_per_replica * num_replicas)

    async def load_and_embed(url, idx):
        async with semaphore:
            print(f"Embedding {url} on replica {idx % num_replicas}")
            embedder_replica = embedder_replicas[idx % num_replicas]
            return await embedder_replica.embed_docs(
                [url], normalize_embeddings=True, run_async=True, stream_logs=False
            )

    start_time = time.time()
    futs = [load_and_embed(url, idx) for idx, url in enumerate(urls)]
    task_results = await tqdm.gather(*futs)

    failures = len([res for res in task_results if isinstance(res, Exception)])
    total_download_time = sum(
        [res[2] for res in task_results if not isinstance(res, Exception)]
    )
    total_embed_time = sum(
        [res[3] for res in task_results if not isinstance(res, Exception)]
    )
    print(
        f"Received {len(task_results) - failures} total embeddings, with {failures} failures.\n"
        f"Embedded {len(urls)} docs across {num_replicas} replicas with {max_concurrency_per_replica} "
        f"concurrent calls: {time.time() - start_time} \n"
        f"Total sys time for downloads: {total_download_time} \n"
        f"Total sys time for embeddings: {total_embed_time}"
    )


# :::note{.info title="Note"}
# Make sure that any code in your Python file that's meant to only run locally runs within
# a `if __name__ == "__main__":` block, as shown below. Otherwise, the script code will run
# when Runhouse attempts to import your code remotely.
# :::
if __name__ == "__main__":
    asyncio.run(main())
