"""
向量数据库工具类
"""
from typing import Optional, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config.settings import VECTOR_DB_CONFIG
from src.utils.embeddings import get_embedding_function


class VectorDBManager:
    """向量数据库管理器"""
    
    def __init__(self, persist_directory: Optional[str] = None, embedding_type: str = "huggingface"):
        self.persist_directory = persist_directory or VECTOR_DB_CONFIG["chroma_persist_dir"]
        self.embedding_function = get_embedding_function(embedding_type)
        self.vectorstore = None
        self.retriever = None
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化向量数据库"""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": VECTOR_DB_CONFIG["retriever_k"]}
            )
            print(f"✅ 向量数据库加载成功: {self.persist_directory}")
        except Exception as e:
            print(f"❌ 向量数据库加载失败: {e}")
            self.vectorstore = None
            self.retriever = None
    
    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """检索相关文档"""
        if self.retriever is None:
            return []
        
        search_k = k or VECTOR_DB_CONFIG["retriever_k"]
        try:
            return self.retriever.invoke(query, k=search_k)
        except Exception as e:
            print(f"❌ 文档检索失败: {e}")
            return []
    
    def format_documents(self, docs: List[Document]) -> str:
        """格式化文档内容"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, query: str, k: Optional[int] = None) -> dict:
        """查询向量数据库并返回格式化结果"""
        docs = self.retrieve_documents(query, k)
        context = self.format_documents(docs)
        return {
            "context": context,
            "documents": docs
        }