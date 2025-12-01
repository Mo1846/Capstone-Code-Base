"""
集成诊断代理
"""
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import API_CONFIG
from src.agents.knowledge_agents import TCMKnowledgeAgent, WMKnowledgeAgent
from src.components.conversation_memory import ConversationMemory
from src.components.diagnostic_questioner import DiagnosticQuestioner


class IntegratedDiagnosticAgent:
    """集成诊断Agent - 整合中西医agent"""
    
    def __init__(self, tcm_database: str = None, wm_persist_dir: str = None):
        self.tcm_agent = TCMKnowledgeAgent(tcm_database)
        self.wm_agent = WMKnowledgeAgent(wm_persist_dir)
        self.conversation_memory = ConversationMemory()
        self.diagnostic_questioner = DiagnosticQuestioner()
        self.is_in_diagnosis_mode = False
        
        # 初始化LLM用于整合结果
        self.integrator_llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0.3,
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"]
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
        diagnosis_keywords = ['诊断', '看病', '症状', '不适', '哪里不舒服', '疼', '痒', '治疗', '病', '问诊']
        return any(keyword in question for keyword in diagnosis_keywords)
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation_memory.history
    
    def reset_conversation(self):
        """重置对话"""
        self.conversation_memory = ConversationMemory()
        self.diagnostic_questioner = DiagnosticQuestioner()
        self.is_in_diagnosis_mode = False
    
    def reset_diagnosis(self):
        """重置诊断模式"""
        self.diagnostic_questioner.reset()
        self.is_in_diagnosis_mode = False
        self.conversation_memory.add_message("system", "诊断模式已重置")