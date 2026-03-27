"""
中西医知识代理
"""
import json
import jieba
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import API_CONFIG
from src.utils.graph_db import GraphDBManager
from src.utils.vector_db import VectorDBManager


class TCMKnowledgeAgent:
    """中医知识Agent"""
    
    def __init__(self, database: str = None):
        # 初始化图数据库连接
        self.graph_manager = GraphDBManager(database)
        self.neo4j_available = self.graph_manager.is_available()
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0,
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"]
        )
        
        if not self.neo4j_available:
            # 如果Neo4j不可用，使用备用方案
            # 从文件加载中医知识
            self.disease_data = self._load_disease_data()
    
    def _load_disease_data(self) -> List[Dict]:
        """加载疾病数据作为备用"""
        try:
            disease_file = "/workspace/tools/disease.jsonl"
            diseases = []
            with open(disease_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        disease = json.loads(line.strip())
                        diseases.append(disease)
                    except:
                        continue
            return diseases
        except:
            return []
    
    def query(self, question: str) -> str:
        """查询中医知识"""
        if self.neo4j_available:
            try:
                return self.graph_manager.query(question)
            except Exception as e:
                return f"中医知识查询出错: {str(e)}"
        else:
            # 使用备用查询方法
            return self._query_from_disease_data(question)
    
    def _query_from_disease_data(self, question: str) -> str:
        """从疾病数据中查询信息"""
        if not self.disease_data:
            return "中医知识库暂时不可用，将使用通用模型进行回答。"
        
        # 简单的关键字匹配
        keywords = jieba.lcut(question)
        # 过滤掉常用词
        keywords = [w for w in keywords if len(w) > 1 and w not in ['的', '了', '在', '是', '我', '有', '和', '就', '都', '而', '及', '与', '或']]
        
        results = []
        for disease in self.disease_data:
            # 检查是否包含关键词
            text = f"{disease.get('name', '')} {disease.get('name_exp', '')} {disease.get('cause', '')} {disease.get('key_point', '')} {disease.get('solution', '')} {disease.get('after', '')}"
            if any(keyword in text for keyword in keywords[:3]):  # 只检查前3个关键词
                results.append({
                    'name': disease.get('name', ''),
                    'description': disease.get('name_exp', '')[:500],  # 限制长度
                    'treatment': disease.get('solution', '')[:500]   # 限制长度
                })
        
        if results:
            response = "根据中医知识库找到以下相关信息：\n\n"
            for i, result in enumerate(results[:3]):  # 只返回前3个结果
                response += f"疾病: {result['name']}\n"
                response += f"描述: {result['description']}\n"
                response += f"治疗: {result['treatment']}\n\n"
            return response
        else:
            return "在中医知识库中未找到相关疾病信息，建议咨询中医专业医师。"


class WMKnowledgeAgent:
    """西医知识Agent"""
    
    def __init__(self, persist_directory: str = None):
        # 初始化西医向量数据库
        self.vector_db = VectorDBManager(persist_directory)
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0.3,
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"]
        )
        
        # 西医回答模板
        self.wm_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。"
             "用户可能是医学生、住院医师、资深中医、普通患者或医学爱好者。"
             "\n- 若问题包含专业术语或机制探讨，可使用规范医学术语，并简要解释关键概念；"
             "\n- 若问题偏向症状描述或日常护理，请用通俗易懂的语言，避免 jargon；"
             "\n- 始终保持尊重、耐心与同理心，不假设、不标签用户身份；"
             "\n- 基于提供的上下文作答，若信息不足，请说明“现有资料较少，我将尽我所能为你解释”，并且根据你的原有知识作答；"
             "\n- 回答需简洁，聚焦核心信息，避免冗长；"
             "\n- 在回答末尾，用开放式提问引导用户深入探讨（如：‘你是否还想了解其鉴别诊断？’ 或 ‘需要我解释治疗方案的细节吗？’）"
            ),
            ("human", "参考资料：\n{context}\n\n用户问题：{question}")
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.wm_chain = (self.wm_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """查询西医知识"""
        try:
            if self.vector_db.retriever is None:
                # 如果没有可用的检索器，使用通用模型回答
                return self._query_with_general_model(question)
            
            result = self.vector_db.query(question)
            context = result["context"]
            
            input_data = {"context": context, "question": question}
            response = self.wm_chain.invoke(input_data)
            
            return response
        except Exception as e:
            return f"西医知识查询出错: {str(e)}"
    
    def _query_with_general_model(self, question: str) -> str:
        """使用通用模型回答问题"""
        # 创建一个简单的提示模板
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。"
             "用户可能是医学生、住院医师、资深中医、普通患者或医学爱好者。"
             "\n- 若问题包含专业术语或机制探讨，可使用规范医学术语，并简要解释关键概念；"
             "\n- 若问题偏向症状描述或日常护理，请用通俗易懂的语言，避免 jargon；"
             "\n- 始终保持尊重、耐心与同理心，不假设、不标签用户身份；"
             "\n- 回答需简洁，聚焦核心信息，避免冗长；"
             "\n- 在回答末尾，用开放式提问引导用户深入探讨（如：‘你是否还想了解其鉴别诊断？’ 或 ‘需要我解释治疗方案的细节吗？’）"
            ),
            ("human", "用户问题：{question}")
        ])
        
        simple_chain = (simple_prompt
            | self.llm
            | StrOutputParser()
        )
        
        try:
            result = simple_chain.invoke({"question": question})
            return result
        except Exception as e:
            return f"西医知识查询出错: {str(e)}"