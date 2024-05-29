
class URLEmbedder:
    def __init__(self, **model_kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(**model_kwargs)

    def load_and_split_doc(self, url: str):
        from langchain_community.document_loaders import WebBaseLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = WebBaseLoader(
            web_paths=[url],
        ).load()
        splits = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(docs)
        splits_as_str = [doc.page_content for doc in splits]
        return splits_as_str

    def embed(self, url: str, **embed_kwargs):
        splits_as_str = self.load_and_split_doc(url)
        embedding = self.model.encode(splits_as_str, **embed_kwargs)
        return splits_as_str, embedding
