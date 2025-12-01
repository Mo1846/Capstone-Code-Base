"""
解释组件 - 提供可解释AI功能
包括：Chain-of-Thought、反事实解释、追问式解释等
"""
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config.settings import API_CONFIG


class ExplanationComponent:
    """解释组件 - 提供各种可解释AI功能"""
    
    def __init__(self):
        # 初始化LLM用于解释生成
        self.explanation_llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0.1,
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"]
        )
        
        # 反事实解释提示模板
        self.counterfactual_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断专家，擅长提供反事实解释。\n"
             "当用户提供'如果...会怎样'的问题时，请分析改变特定症状或条件后对诊断和治疗的影响。\n"
             "请明确说明改变的影响，并对比原始情况。"
            ),
            ("human", 
             "原始症状: {original_symptoms}\n"
             "原始诊断: {original_diagnosis}\n"
             "反事实条件: {counterfactual_condition}\n"
             "请分析如果出现这种反事实条件，诊断和治疗建议会如何变化。"
            )
        ])
        
        # 依据溯源提示模板
        self.citation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断专家，需要为诊断和治疗建议提供可靠的依据。\n"
             "请引用相关的医学典籍、指南或研究作为支撑，格式为：[依据来源]。"
            ),
            ("human", 
             "诊断结论: {diagnosis}\n"
             "治疗建议: {treatment}\n"
             "请为以上诊断和治疗提供可靠的医学依据。"
            )
        ])
        
        # 对比解释提示模板
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断专家，擅长提供对比解释。\n"
             "请解释为什么选择当前方案而非其他可能的方案，突出关键区别点。"
            ),
            ("human", 
             "当前方案: {current_approach}\n"
             "替代方案: {alternative_approach}\n"
             "患者情况: {patient_context}\n"
             "请解释为什么选择当前方案而不是替代方案。"
            )
        ])
        
        # 交互式追问提示模板
        self.interactive_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断专家，能够回答用户对诊断过程的追问。\n"
             "请基于之前的诊断推理过程，详细回答用户的问题。"
            ),
            ("human", 
             "原始诊断: {original_diagnosis}\n"
             "诊断推理过程: {reasoning_process}\n"
             "用户问题: {user_question}\n"
             "请回答用户关于诊断过程的问题。"
            )
        ])
        
        # 构建解释链
        self.counterfactual_chain = (self.counterfactual_prompt 
                                   | self.explanation_llm 
                                   | StrOutputParser())
        self.citation_chain = (self.citation_prompt 
                             | self.explanation_llm 
                             | StrOutputParser())
        self.comparison_chain = (self.comparison_prompt 
                               | self.explanation_llm 
                               | StrOutputParser())
        self.interactive_chain = (self.interactive_prompt 
                                | self.explanation_llm 
                                | StrOutputParser())
    
    def generate_counterfactual_explanation(self, original_symptoms: str, 
                                          original_diagnosis: str, 
                                          counterfactual_condition: str) -> str:
        """生成反事实解释"""
        try:
            result = self.counterfactual_chain.invoke({
                "original_symptoms": original_symptoms,
                "original_diagnosis": original_diagnosis,
                "counterfactual_condition": counterfactual_condition
            })
            return result
        except Exception as e:
            return f"生成反事实解释时出现错误: {str(e)}"
    
    def generate_citation_explanation(self, diagnosis: str, treatment: str) -> str:
        """生成依据溯源解释"""
        try:
            result = self.citation_chain.invoke({
                "diagnosis": diagnosis,
                "treatment": treatment
            })
            return result
        except Exception as e:
            return f"生成依据溯源解释时出现错误: {str(e)}"
    
    def generate_comparison_explanation(self, current_approach: str, 
                                      alternative_approach: str, 
                                      patient_context: str) -> str:
        """生成对比解释"""
        try:
            result = self.comparison_chain.invoke({
                "current_approach": current_approach,
                "alternative_approach": alternative_approach,
                "patient_context": patient_context
            })
            return result
        except Exception as e:
            return f"生成对比解释时出现错误: {str(e)}"
    
    def generate_interactive_explanation(self, original_diagnosis: str, 
                                       reasoning_process: str, 
                                       user_question: str) -> str:
        """生成交互式追问解释"""
        try:
            result = self.interactive_chain.invoke({
                "original_diagnosis": original_diagnosis,
                "reasoning_process": reasoning_process,
                "user_question": user_question
            })
            return result
        except Exception as e:
            return f"生成交互式解释时出现错误: {str(e)}"
    
    def is_counterfactual_query(self, question: str) -> bool:
        """判断是否为反事实查询"""
        counterfactual_keywords = ['如果', '假如', '要是', '假设', '万一', '若']
        return any(keyword in question for keyword in counterfactual_keywords)
    
    def is_interactive_query(self, question: str, conversation_context: str = "") -> bool:
        """判断是否为交互式追问"""
        interactive_keywords = ['为什么', '如何', '怎么', '原因', '依据', '区别', '对比', '解释']
        has_context_reference = any(word in conversation_context for word in ['诊断', '分析', '建议', '治疗', '方案'])
        return any(keyword in question for keyword in interactive_keywords) and has_context_reference