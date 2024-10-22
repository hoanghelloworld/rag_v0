from model_setup import load_model
from pipeline_setup import create_text_generation_pipeline
from document_processing import load_and_split_documents, create_embeddings, index_documents_elasticsearch
from qa_system import setup_elasticsearch_connection, test_qa_system_elasticsearch

base_model_path = "/llama-70b-instruct"
pdf_path = "/tuyen_sinh.pdf"
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
index_name = "document_index"

model, tokenizer = load_model(base_model_path)

query_pipeline = create_text_generation_pipeline(model, tokenizer)

documents = load_and_split_documents(pdf_path)

embeddings = create_embeddings(embedding_model_name)

es_client = setup_elasticsearch_connection()


index_documents_elasticsearch(es_client, index_name, documents, embeddings)


test_query = "nam nay nhà trường lấy bao nhiêu chỉ tiêu tuyển sinh?"
test_qa_system_elasticsearch(es_client, index_name, embeddings, test_query)
