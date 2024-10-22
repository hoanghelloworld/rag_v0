from elasticsearch import Elasticsearch
import numpy as np

def setup_elasticsearch_connection(host="localhost", port=9200):
    """
    Connect to Elasticsearch.
    """
    es_client = Elasticsearch([{"host": host, "port": port}])
    if not es_client.ping():
        raise ValueError("Connection to Elasticsearch failed!")
    print("Connected to Elasticsearch.")
    return es_client

def search_elasticsearch(es_client, index_name, query_embedding, top_k=5):
    """
    Search for the most similar documents in Elasticsearch using the query embedding.
    """
    query_body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding},
                },
            }
        },
    }
    response = es_client.search(index=index_name, body=query_body)
    return response["hits"]["hits"]

def test_qa_system_elasticsearch(es_client, index_name, embeddings, query):
    """
    Perform QA using Elasticsearch.
    """
    print(f"Query: {query}\n")
    query_embedding = embeddings.embed_query(query)

    results = search_elasticsearch(es_client, index_name, query_embedding)
    
    print(f"Retrieved {len(results)} documents.")
    for result in results:
        print(f"Score: {result['_score']}")
        print(f"Content: {result['_source']['content']}\n")
