from langchain_huggingface import HuggingFaceEmbeddings

class E5Embeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts):
        texts = [f"passage: {text}" for text in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        return super().embed_query(f"query: {text}")
