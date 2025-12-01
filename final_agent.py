#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中西医结合诊疗系统 - Final Agent
功能：
1. 整合中西医agent的输出结果
2. 实现对话记忆功能
3. 实现逐步问诊功能
"""

import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ConversationMemory:
    """对话记忆类"""
    def __init__(self):
        self.history = []
        self.patient_info = {}
        self.diagnosis_process = []
        
    def add_message(self, role: str, content: str):
        """添加对话记录"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        })
    
    def update_patient_info(self, info: Dict):
        """更新患者信息"""
        self.patient_info.update(info)
    
    def add_diagnosis_step(self, step: str, thought: str, action: str):
        """添加诊断步骤"""
        self.diagnosis_process.append({
            "step": step,
            "thought": thought,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_context(self) -> str:
        """获取上下文信息"""
        context_parts = []
        
        # 添加患者信息
        if self.patient_info:
            context_parts.append("患者信息:")
            for key, value in self.patient_info.items():
                context_parts.append(f"  {key}: {value}")
        
        # 添加诊断过程
        if self.diagnosis_process:
            context_parts.append("\n诊断过程:")
            for i, step in enumerate(self.diagnosis_process, 1):
                context_parts.append(f"  步骤{i}: {step['step']}")
                context_parts.append(f"    思考: {step['thought']}")
                context_parts.append(f"    行动: {step['action']}")
        
        # 添加最近的对话历史
        if self.history:
            context_parts.append("\n最近对话:")
            for msg in self.history[-5:]:  # 只取最近5条消息
                context_parts.append(f"  {msg['role']}: {msg['content']}")
        
        return "\n".join(context_parts)


class TCMKnowledgeAgent:
    """中医知识Agent"""
    def __init__(self):
        # 初始化Neo4j图数据库连接
        try:
            self.graph = Neo4jGraph(database='neo4j-2025-10-27t07-22-12')
            self.neo4j_available = True
        except Exception as e:
            print(f"Neo4j连接失败: {e}")
            print("将使用替代方案进行中医知识查询")
            self.neo4j_available = False
            self.graph = None
        
        # 使用OpenAI兼容接口替代ChatTongyi
        self.llm = ChatOpenAI(
            model="qwen-max",
            temperature=0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-db85459561d04810ac504107dbd02936"
        )
        
        if self.neo4j_available:
            # Cypher查询模板
            CYPHER_GENERATION_TEMPLATE = """
            你是一个 Neo4j 专家，任务是将自然语言问题转换为 Cypher 查询。
            图数据库的 schema 如下：
            {schema}

            请严格遵守以下规则：
            1. **节点匹配必须使用 `id` 属性进行精确匹配**，例如：`(n:皮肤病 {{id: "扁平疣"}})`
            2. 不要使用 `CONTAINS`、`=~` 或其他模糊匹配。
            3. 只返回 Cypher 查询语句，不要解释，不要 markdown，不要反引号。
            4. 对于病症遵循以下规则：皮肤病-[辨证为]->证型-[主症包括]->症状
            5. 证型-[治法为]->方剂-[用于治疗]->皮肤病
            问题：{question}
            """
            
            cypher_prompt = PromptTemplate(
                template=CYPHER_GENERATION_TEMPLATE,
                input_variables=["schema", "question"]
            )
            
            self.chain = GraphCypherQAChain.from_llm(
                graph=self.graph,
                llm=self.llm,
                cypher_prompt=cypher_prompt,  
                verbose=True,
                allow_dangerous_requests=True,
                validate_cypher=True,
                fix_cypher=True, 
                max_fix_attempts=2,
            )
        else:
            # 如果Neo4j不可用，使用备用方案
            # 从文件加载中医知识
            self.disease_data = self._load_disease_data()
    
    def _load_disease_data(self):
        """加载疾病数据作为备用"""
        try:
            disease_file = "/workspace/tools/disease.jsonl"
            import json
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
                response = self.chain.invoke({"query": question})
                return response.get("result", "未找到相关信息")
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
        import jieba
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
    def __init__(self):
        # 初始化西医向量数据库 - 如果无法加载HuggingFaceEmbeddings，则使用OpenAI嵌入
        try:
            # 尝试使用HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            print(f"HuggingFaceEmbeddings加载失败: {e}")
            print("将使用OpenAI嵌入作为替代方案")
            # 使用OpenAI的嵌入作为替代
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key="sk-db85459561d04810ac504107dbd02936",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        
        try:
            self.vectorstore = Chroma(
                persist_directory="./basic app/chroma_db",
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            print(f"向量数据库加载失败: {e}")
            print("将使用简单的替代方案")
            # 创建一个简单的替代方案
            self.vectorstore = None
            self.retriever = None
        
        # 初始化LLM
        self.llm = ChatOpenAI(
            model="qwen-max",
            temperature=0.3,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-db85459561d04810ac504107dbd02936"
        )
        
        # 西医回答模板
        self.wm_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。"
             "用户可能是医学生、住院医师、资深中医、普通患者或医学爱好者。"
             "请根据问题中隐含的专业程度，自动调整回答的深度与术语使用："
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
            if self.retriever is None:
                # 如果没有可用的检索器，使用通用模型回答
                return self._query_with_general_model(question)
            
            retrieved_docs = self.retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            input_data = {"context": context, "question": question}
            result = self.wm_chain.invoke(input_data)
            
            return result
        except Exception as e:
            return f"西医知识查询出错: {str(e)}"
    
    def _query_with_general_model(self, question: str) -> str:
        """使用通用模型回答问题"""
        # 创建一个简单的提示模板
        simple_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一位经验丰富的皮肤科临床辅助诊疗专家，正在辅助不同背景的用户理解皮肤病相关知识。"
             "用户可能是医学生、住院医师、资深中医、普通患者或医学爱好者。"
             "请根据问题中隐含的专业程度，自动调整回答的深度与术语使用："
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


class DiagnosticQuestioner:
    """诊断询问器 - 实现逐步问诊功能"""
    def __init__(self):
        self.question_templates = [
            "请问您主要的不适症状是什么？",
            "这些症状持续多长时间了？",
            "症状是持续存在还是间歇性出现？",
            "有没有什么因素会加重或缓解您的症状？",
            "您以前是否出现过类似的情况？",
            "您目前是否在使用任何药物治疗？",
            "您是否有过敏史？",
            "您的家族中是否有类似的疾病史？",
            "除了主要症状外，还有没有其他伴随症状？",
            "您最近的生活作息和精神状态如何？",
            "您平时的饮食习惯如何？",
            "您是否有其他慢性疾病？"
        ]
        
        # 问诊状态
        self.current_question_index = 0
        self.collected_info = {}
        self.is_diagnosis_complete = False
    
    def get_next_question(self, conversation_memory: ConversationMemory) -> str:
        """根据当前收集的信息获取下一个问题"""
        # 如果诊断已完成，返回结束语
        if self.is_diagnosis_complete:
            return "感谢您的配合，我们已经收集了足够的信息进行初步诊断。"
        
        # 检查是否还有预设问题
        if self.current_question_index < len(self.question_templates):
            question = self.question_templates[self.current_question_index]
            self.current_question_index += 1
            return question
        else:
            # 所有预设问题问完，结束问诊
            self.is_diagnosis_complete = True
            return "感谢您的配合，我们已经收集了足够的信息进行初步诊断。"
    
    def process_answer(self, answer: str, conversation_memory: ConversationMemory):
        """处理用户回答并更新对话记忆"""
        # 这里可以添加逻辑来解析用户回答并提取关键信息
        # 暂时简单记录
        conversation_memory.add_message("user", answer)
        
        # 简单的关键词提取
        keywords = self.extract_keywords(answer)
        if keywords:
            conversation_memory.update_patient_info({"symptoms_keywords": keywords})
    
    def extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 简单的关键词提取逻辑，实际应用中可以使用更复杂的NLP技术
        import jieba
        # 这里只做示例，实际应该根据医学术语库来提取
        words = jieba.lcut(text)
        # 简单过滤，保留长度大于1的词
        keywords = [w for w in words if len(w) > 1 and w not in ['的', '了', '在', '是', '我', '有', '和', '就', '都', '而', '及', '与', '或']]
        return keywords[:10]  # 返回前10个关键词


class IntegratedDiagnosticAgent:
    """集成诊断Agent - 整合中西医agent"""
    def __init__(self):
        self.tcm_agent = TCMKnowledgeAgent()
        self.wm_agent = WMKnowledgeAgent()
        self.conversation_memory = ConversationMemory()
        self.diagnostic_questioner = DiagnosticQuestioner()
        self.is_in_diagnosis_mode = False
        
        # 初始化LLM用于整合结果
        self.integrator_llm = ChatOpenAI(
            model="qwen-max",
            temperature=0.3,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-db85459561d04810ac504107dbd02936"
        )
        
        # 整合结果的提示模板
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是中西医结合诊疗专家，需要将中医和西医的诊断信息进行整合分析。"
             "请遵循以下原则：\n"
             "1. 客观呈现中西医各自的观点和建议\n"
             "2. 寻找中西医理论的结合点\n"
             "3. 提出中西医结合的诊疗建议\n"
             "4. 保持专业、准确、易懂的表达方式\n"
             "5. 在适当时候建议患者寻求专业医生的进一步诊断"
            ),
            ("human", 
             "中医诊断信息：\n{tcm_info}\n\n"
             "西医诊断信息：\n{wm_info}\n\n"
             "上下文信息：\n{context}\n\n"
             "请整合以上信息，给出中西医结合的分析结果。"
            )
        ])
        
        self.integration_chain = (self.integration_prompt
            | self.integrator_llm
            | StrOutputParser()
        )
    
    def start_diagnosis_mode(self):
        """开始诊断模式"""
        self.is_in_diagnosis_mode = True
        self.conversation_memory.add_message("system", "已进入逐步问诊模式")
        return "您好，现在开始进行逐步问诊。我将根据您的症状逐步了解情况，以便给出更准确的建议。"
    
    def continue_diagnosis(self, user_response: str) -> str:
        """继续诊断流程"""
        # 处理用户回应
        self.diagnostic_questioner.process_answer(user_response, self.conversation_memory)
        
        # 获取下一个问题
        next_question = self.diagnostic_questioner.get_next_question(self.conversation_memory)
        
        # 记录系统消息
        self.conversation_memory.add_message("assistant", next_question)
        
        return next_question
    
    def query(self, question: str) -> str:
        """处理查询请求"""
        # 记录用户问题
        self.conversation_memory.add_message("user", question)
        
        # 检查是否需要启动诊断模式
        if self.should_start_diagnosis(question):
            response = self.start_diagnosis_mode()
            next_question = self.diagnostic_questioner.get_next_question(self.conversation_memory)
            self.conversation_memory.add_message("assistant", next_question)
            return f"{response}\n{next_question}"
        
        # 检查是否在诊断模式下
        if self.is_in_diagnosis_mode and not self.diagnostic_questioner.is_diagnosis_complete:
            return self.continue_diagnosis(question)
        
        # 正常查询模式 - 获取中西医信息
        tcm_result = self.tcm_agent.query(question)
        wm_result = self.wm_agent.query(question)
        
        # 整合结果
        context = self.conversation_memory.get_context()
        
        try:
            integration_result = self.integration_chain.invoke({
                "tcm_info": tcm_result,
                "wm_info": wm_result,
                "context": context
            })
        except Exception as e:
            integration_result = f"整合结果时出现错误: {str(e)}\n\n中医信息: {tcm_result}\n\n西医信息: {wm_result}"
        
        # 记录思考过程
        self.conversation_memory.add_diagnosis_step(
            step="信息整合",
            thought=f"整合中西医诊断信息，生成综合建议",
            action=f"查询问题: {question}"
        )
        
        # 记录系统回复
        self.conversation_memory.add_message("assistant", integration_result)
        
        return integration_result
    
    def should_start_diagnosis(self, question: str) -> bool:
        """判断是否应该启动诊断模式"""
        # 检查是否是诊断相关问题
        diagnosis_keywords = ['诊断', '看病', '症状', '不适', '哪里不舒服', '疼', '痒', '治疗', '病']
        return any(keyword in question for keyword in diagnosis_keywords) and '问诊' in question
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_memory.history
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_memory = ConversationMemory()
        self.diagnostic_questioner = DiagnosticQuestioner()
        self.is_in_diagnosis_mode = False


def main():
    """主函数 - 交互式界面"""
    agent = IntegratedDiagnosticAgent()
    
    print("=" * 60)
    print("中西医结合诊疗系统 - Final Agent")
    print("=" * 60)
    print("功能说明：")
    print("1. 智能回答中西医相关问题")
    print("2. 支持逐步问诊功能")
    print("3. 整合中西医诊断信息")
    print("4. 记忆对话历史和患者信息")
    print("输入 'quit' 或 'exit' 退出系统")
    print("输入 'history' 查看对话历史")
    print("输入 'reset' 重置对话")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n请输入您的问题或症状描述: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("感谢使用中西医结合诊疗系统，再见！")
                break
            elif user_input.lower() == 'history':
                history = agent.get_conversation_history()
                print(f"\n对话历史 (共{len(history)}条):")
                for i, msg in enumerate(history, 1):
                    print(f"  {i}. [{msg['timestamp']}] {msg['role']}: {msg['content'][:50]}...")
                continue
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                print("对话已重置")
                continue
            elif not user_input:
                continue
            
            print("正在分析中，请稍候...")
            response = agent.query(user_input)
            print(f"\n系统回复: {response}")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            print("请重新输入或联系技术支持")


if __name__ == "__main__":
    main()