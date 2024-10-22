from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def load_and_split_documents(pdf_path, chunk_size=1000, chunk_overlap=20):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(documents)
    return all_splits

def create_embeddings(model_name, device="cuda"):
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return embeddings

def index_documents_elasticsearch(es_client, index_name, documents, embeddings):
    """
    Index the documents into Elasticsearch with embeddings.
    """
    actions = []
    for i, doc in enumerate(documents):
        embedding = embeddings.embed_query(doc.page_content)
        action = {
            "_index": index_name,
            "_id": i,
            "_source": {
                "content": doc.page_content,
                "embedding": embedding,
                "metadata": doc.metadata,
            },
        }
        actions.append(action)
    
    bulk(es_client, actions)
    print(f"Indexed {len(documents)} documents into Elasticsearch.")
