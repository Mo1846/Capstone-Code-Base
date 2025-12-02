"""
诊断询问器组件
"""
from typing import List
from src.components.conversation_memory import ConversationMemory


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
    
    def reset(self):
        """重置问诊状态"""
        self.current_question_index = 0
        self.collected_info = {}
        self.is_diagnosis_complete = False