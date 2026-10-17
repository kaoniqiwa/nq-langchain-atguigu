# Milvus 的 Python 客户端。这里用的是 MilvusClient——pymilvus 2.4+ 的新式 API，
# 内部自己管连接池，不需要老 API 那套 connections.connect() / disconnect()。
# 19530 是 Milvus 服务端默认的 gRPC 端口（单机和集群都是它），
# 前提是本机已经把服务起起来了，例如官方 standalone 的 compose 文件：
#     docker compose -f milvus-standalone-docker-compose.yml up -d
# 服务没起的话，这一行未必报错——gRPC 的通道是懒建立的，
# 通常要等到下面第一次真正发请求才抛连接失败。所以遇到连接类报错，
# 先确认服务在不在，再怀疑这段代码
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# database 是 collection 之上的一层隔离：一个库里可以有多个 collection，
# 不同库之间的同名 collection 互不干扰。再往上还有一层可选的 partition，
# 用来在单个 collection 内按业务维度（如日期、租户）切分数据。
# 单机部署默认只有一个 'default' 库，且它删不掉
db_name = 'rag_demo'
existing_dbs = client.list_databases()
# 先查再建：create_database 对已存在的库名会直接抛错，不是幂等操作。
# 加上这个判断，脚本才能反复运行，不用每次手动清理现场
if db_name not in existing_dbs:
    client.create_database(db_name)

print(client.list_databases())  # ['default', 'rag_demo']

client.use_database(db_name)
# 删库会把库里所有 collection 连同向量数据一起删掉，不可恢复。
# 放在最后是为了让脚本跑完不留痕，下次运行仍是干净状态；
# 想留着数据做后续实验（比如接着做建 collection、插数据），把下面两行注释掉
client.drop_database(db_name)
print(client.list_databases())  # ['default']
