"""
对话记忆组件
"""
from typing import Dict, List, Any
from datetime import datetime


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
    
    def clear(self):
        """清空记忆"""
        self.history = []
        self.patient_info = {}
        self.diagnosis_process = []