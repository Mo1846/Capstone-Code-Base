"""
配置文件 - 存储项目配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API配置
API_CONFIG = {
    "model_name": "qwen-max",
    "temperature": 0.3,
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("DASHSCOPE_API_KEY", "sk-db85459561d04810ac504107dbd02936"),
    "neo4j_database": "neo4j-2025-10-27t07-22-12"
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "huggingface_model": "BAAI/bge-small-zh-v1.5",
    "openai_model": "text-embedding-ada-002",
    "dashscope_model": "text-embedding-v2"
}

# 向量数据库配置
VECTOR_DB_CONFIG = {
    "chroma_persist_dir": "./basic app/chroma_db",
    "retriever_k": 5
}

# 系统配置
SYSTEM_CONFIG = {
    "tokenizers_parallelism": "false",
    "max_fix_attempts": 2
}