"""
集成诊断代理 - 支持可解释AI
"""
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.config.settings import API_CONFIG
from src.agents.knowledge_agents import TCMKnowledgeAgent, WMKnowledgeAgent
from src.components.conversation_memory import ConversationMemory
from src.components.diagnostic_questioner import DiagnosticQuestioner
from src.components.explanation_component import ExplanationComponent
import json


class IntegratedDiagnosticAgent:
    """集成诊断Agent - 整合中西医agent"""
    
    def __init__(self, tcm_database: str = None, wm_persist_dir: str = None):
        self.tcm_agent = TCMKnowledgeAgent(tcm_database)
        self.wm_agent = WMKnowledgeAgent(wm_persist_dir)
        self.conversation_memory = ConversationMemory()
        self.diagnostic_questioner = DiagnosticQuestioner()
        self.explanation_component = ExplanationComponent()  # 新增解释组件
        self.is_in_diagnosis_mode = False
        
        # 初始化LLM用于整合结果
        self.integrator_llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0.3,
            base_url=API_CONFIG["base_url"],
            api_key=API_CONFIG["api_key"]
        )
        
        # 初始化用于可解释性的LLM（更低的temperature以获得更一致的解释）
        self.explanation_llm = ChatOpenAI(
            model=API_CONFIG["model_name"],
            temperature=0.1,
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
        
        # Chain-of-Thought推理模板
        self.cot_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断推理专家。请使用Chain-of-Thought方法逐步分析问题。\n"
             "输出格式要求：\n"
             "步骤1: [分析中医信息]\n"
             "步骤2: [分析西医信息]\n"
             "步骤3: [对比分析]\n"
             "步骤4: [整合结论]\n"
             "参考文献: [如有引用请说明]\n\n"
             "请确保每一步都有明确的逻辑推理过程。"
            ),
            ("human", 
             "中医诊断信息：\n{tcm_info}\n\n"
             "西医诊断信息：\n{wm_info}\n\n"
             "上下文信息：\n{context}\n\n"
             "请逐步分析并给出中西医结合的诊断结论。"
            )
        ])
        
        # 结构化输出模板（JSON格式）
        self.json_integration_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的中西医结合诊断专家。请按照以下JSON格式输出诊断结果：\n"
             "{\n"
             "  \"diagnosis\": \"诊断结论\",\n"
             "  \"reasoning\": [\n"
             "    \"推理步骤1\",\n"
             "    \"推理步骤2\",\n"
             "    \"推理步骤3\"\n"
             "  ],\n"
             "  \"tcm_analysis\": \"中医分析\",\n"
             "  \"wm_analysis\": \"西医分析\",\n"
             "  \"recommendation\": \"治疗建议\",\n"
             "  \"confidence\": \"置信度(1-10)\",\n"
             "  \"reference\": \"参考文献或依据\"\n"
             "}"
            ),
            ("human", 
             "中医诊断信息：\n{tcm_info}\n\n"
             "西医诊断信息：\n{wm_info}\n\n"
             "上下文信息：\n{context}\n\n"
             "请按JSON格式输出中西医结合的分析结果。"
            )
        ])
        
        self.integration_chain = (self.integration_prompt
            | self.integrator_llm
            | StrOutputParser()
        )
        
        self.cot_chain = (self.cot_prompt
            | self.integrator_llm
            | StrOutputParser()
        )
        
        # JSON输出解析器
        self.json_parser = JsonOutputParser()
        self.json_integration_chain = (self.json_integration_prompt
            | self.explanation_llm
            | StrOutputParser()
        )
        
        # 解释偏好设置
        self.explanation_preference = "detailed"  # 默认详细解释
    
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
        # 检查是否是解释偏好设置指令
        if question.startswith('/explain'):
            return self.set_explanation_preference(question)
        
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
        
        # 检查是否为反事实查询
        if self.explanation_component.is_counterfactual_query(question):
            # 需要获取之前的诊断信息
            context = self.conversation_memory.get_context()
            if "诊断" in context or "分析" in context:
                # 提取之前的诊断信息
                prev_diagnosis = self._extract_previous_diagnosis(context)
                if prev_diagnosis:
                    counterfactual_result = self.explanation_component.generate_counterfactual_explanation(
                        original_symptoms="用户之前的症状描述",  # 实际应用中需要从上下文提取
                        original_diagnosis=prev_diagnosis,
                        counterfactual_condition=question
                    )
                    # 记录系统回复
                    self.conversation_memory.add_message("assistant", counterfactual_result)
                    return counterfactual_result
        
        # 检查是否为交互式追问
        context = self.conversation_memory.get_context()
        if self.explanation_component.is_interactive_query(question, context):
            # 提取之前的诊断和推理过程
            prev_diagnosis = self._extract_previous_diagnosis(context)
            reasoning_process = self._extract_reasoning_process(context)
            if prev_diagnosis or reasoning_process:
                interactive_result = self.explanation_component.generate_interactive_explanation(
                    original_diagnosis=prev_diagnosis or "未找到之前的诊断",
                    reasoning_process=reasoning_process or "未找到推理过程",
                    user_question=question
                )
                # 记录系统回复
                self.conversation_memory.add_message("assistant", interactive_result)
                return interactive_result
        
        # 正常查询模式 - 获取中西医信息
        tcm_result = self.tcm_agent.query(question)
        wm_result = self.wm_agent.query(question)
        
        # 整合结果
        context = self.conversation_memory.get_context()
        
        try:
            # 根据解释偏好选择不同的处理方式
            if self.explanation_preference == "detailed":
                integration_result = self.integration_chain.invoke({
                    "tcm_info": tcm_result,
                    "wm_info": wm_result,
                    "context": context
                })
            elif self.explanation_preference == "cot":
                # 使用Chain-of-Thought推理
                integration_result = self.cot_chain.invoke({
                    "tcm_info": tcm_result,
                    "wm_info": wm_result,
                    "context": context
                })
            elif self.explanation_preference == "structured":
                # 使用结构化JSON输出
                try:
                    json_result = self.json_integration_chain.invoke({
                        "tcm_info": tcm_result,
                        "wm_info": wm_result,
                        "context": context
                    })
                    # 尝试解析JSON并格式化输出
                    parsed_result = self._parse_json_result(json_result)
                    integration_result = self._format_structured_output(parsed_result)
                except:
                    # 如果JSON解析失败，回退到普通整合
                    integration_result = self.integration_chain.invoke({
                        "tcm_info": tcm_result,
                        "wm_info": wm_result,
                        "context": context
                    })
            elif self.explanation_preference == "brief":
                # 简洁模式，直接整合结果
                integration_result = self.integration_chain.invoke({
                    "tcm_info": tcm_result,
                    "wm_info": wm_result,
                    "context": context
                })
                # 简化输出
                integration_result = self._simplify_output(integration_result)
            else:
                # 默认模式
                integration_result = self.integration_chain.invoke({
                    "tcm_info": tcm_result,
                    "wm_info": wm_result,
                    "context": context
                })
        except Exception as e:
            integration_result = f"整合结果时出现错误: {str(e)}\n\n中医信息: {tcm_result}\n\n西医信息: {wm_result}"
        
        # 生成依据溯源解释（如果需要）
        if self.explanation_preference in ["detailed", "structured"]:
            try:
                citation_result = self.explanation_component.generate_citation_explanation(
                    diagnosis=integration_result[:200],  # 限制长度
                    treatment="治疗建议部分"
                )
                integration_result += f"\n\n【医学依据】\n{citation_result}"
            except:
                pass  # 如果生成依据失败，不影响主要结果
        
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
    
    def set_explanation_preference(self, command: str) -> str:
        """设置解释偏好"""
        command_parts = command.split()
        if len(command_parts) < 2:
            return f"当前解释模式: {self.explanation_preference}\n可用模式: detailed, brief, cot (Chain-of-Thought), structured"
        
        mode = command_parts[1].lower()
        if mode in ['detailed', 'brief', 'cot', 'structured']:
            self.explanation_preference = mode
            return f"解释模式已设置为: {mode}\n- detailed: 详细解释\n- brief: 简洁解释\n- cot: Chain-of-Thought推理\n- structured: 结构化JSON输出"
        else:
            return f"无效的解释模式: {mode}\n可用模式: detailed, brief, cot, structured"
    
    def _extract_previous_diagnosis(self, context: str) -> str:
        """从上下文提取之前的诊断信息"""
        # 简单的提取方法，实际应用中可能需要更复杂的逻辑
        lines = context.split('\n')
        diagnosis_lines = []
        for line in lines:
            if '诊断' in line or '分析' in line or '结论' in line:
                diagnosis_lines.append(line.strip())
        return ' '.join(diagnosis_lines) if diagnosis_lines else ""
    
    def _extract_reasoning_process(self, context: str) -> str:
        """从上下文提取推理过程"""
        # 简单的提取方法，实际应用中可能需要更复杂的逻辑
        lines = context.split('\n')
        reasoning_lines = []
        for line in lines:
            if '步骤' in line or '推理' in line or '原因' in line or '思考' in line:
                reasoning_lines.append(line.strip())
        return ' '.join(reasoning_lines) if reasoning_lines else ""
    
    def _parse_json_result(self, json_str: str) -> dict:
        """解析JSON结果"""
        try:
            # 尝试从代码块中提取JSON
            if "```json" in json_str:
                start = json_str.find("```json") + 7
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.find("```") + 3
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            
            return json.loads(json_str)
        except:
            # 如果解析失败，尝试直接解析原始字符串
            try:
                return json.loads(json_str)
            except:
                # 如果还是失败，返回原始字符串
                return {"raw_output": json_str}
    
    def _format_structured_output(self, parsed_result: dict) -> str:
        """格式化结构化输出"""
        if "raw_output" in parsed_result:
            return parsed_result["raw_output"]
        
        formatted_parts = []
        
        if "diagnosis" in parsed_result:
            formatted_parts.append(f"【诊断结论】\n{parsed_result['diagnosis']}")
        
        if "reasoning" in parsed_result:
            formatted_parts.append(f"【推理过程】")
            for i, step in enumerate(parsed_result['reasoning'], 1):
                formatted_parts.append(f"  步骤{i}: {step}")
        
        if "tcm_analysis" in parsed_result:
            formatted_parts.append(f"【中医分析】\n{parsed_result['tcm_analysis']}")
        
        if "wm_analysis" in parsed_result:
            formatted_parts.append(f"【西医分析】\n{parsed_result['wm_analysis']}")
        
        if "recommendation" in parsed_result:
            formatted_parts.append(f"【治疗建议】\n{parsed_result['recommendation']}")
        
        if "confidence" in parsed_result:
            formatted_parts.append(f"【置信度】\n{parsed_result['confidence']}/10")
        
        if "reference" in parsed_result:
            formatted_parts.append(f"【参考依据】\n{parsed_result['reference']}")
        
        return "\n\n".join(formatted_parts)
    
    def _simplify_output(self, text: str) -> str:
        """简化输出"""
        lines = text.split('\n')
        simplified_lines = []
        for line in lines:
            # 移除过多的空行和格式
            stripped = line.strip()
            if stripped:
                simplified_lines.append(stripped)
        
        result = '\n'.join(simplified_lines)
        # 限制输出长度
        if len(result) > 500:
            result = result[:500] + "...(内容已简化)"
        
        return result
    
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