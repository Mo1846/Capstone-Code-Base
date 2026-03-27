"""
嵌入模型工具类
"""
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from src.config.settings import EMBEDDING_CONFIG, API_CONFIG


class EmbeddingFactory:
    """嵌入模型工厂类，用于创建不同类型的嵌入模型"""
    
    @staticmethod
    def create_huggingface_embedding() -> HuggingFaceEmbeddings:
        """创建HuggingFace嵌入模型"""
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_CONFIG["huggingface_model"],
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            print(f"HuggingFaceEmbeddings加载失败: {e}")
            return EmbeddingFactory.create_openai_embedding()
    
    @staticmethod
    def create_openai_embedding() -> OpenAIEmbeddings:
        """创建OpenAI嵌入模型"""
        try:
            return OpenAIEmbeddings(
                model=EMBEDDING_CONFIG["openai_model"],
                api_key=API_CONFIG["api_key"],
                base_url=API_CONFIG["base_url"]
            )
        except Exception as e:
            print(f"OpenAIEmbeddings加载失败: {e}")
            return EmbeddingFactory.create_dashscope_embedding()
    
    @staticmethod
    def create_dashscope_embedding() -> DashScopeEmbeddings:
        """创建DashScope嵌入模型"""
        try:
            return DashScopeEmbeddings(model=EMBEDDING_CONFIG["dashscope_model"])
        except Exception as e:
            print(f"DashScopeEmbeddings加载失败: {e}")
            raise e
    
    @classmethod
    def create_embedding(cls, preferred_type: str = "huggingface") -> object:
        """根据偏好类型创建嵌入模型，支持fallback机制"""
        if preferred_type == "huggingface":
            return cls.create_huggingface_embedding()
        elif preferred_type == "openai":
            return cls.create_openai_embedding()
        elif preferred_type == "dashscope":
            return cls.create_dashscope_embedding()
        else:
            # 默认使用HuggingFace，失败后自动fallback
            return cls.create_huggingface_embedding()


def get_embedding_function(preferred_type: str = "huggingface"):
    """获取嵌入函数的便捷方法"""
    return EmbeddingFactory.create_embedding(preferred_type)