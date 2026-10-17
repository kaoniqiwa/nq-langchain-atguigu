from pymilvus import MilvusClient
from pymilvus.client.types import MetricType

client = MilvusClient(uri="http://localhost:19530")

db_name = 'rag_demo'
existing_dbs = client.list_databases()
if db_name not in existing_dbs:
    client.create_database(db_name)

client.use_database(db_name)

print(client.list_collections())

collection_name = 'test_collection'
client.create_collection(collection_name, dimension=1024, metric_type='COSINE')
print(client.list_collections())
