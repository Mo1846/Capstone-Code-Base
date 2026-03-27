#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可解释AI功能演示
展示基于API的可解释AI实现，包括Chain-of-Thought推理、反事实解释、追问式解释等功能
"""
import sys
import os
sys.path.append('/workspace')

from src.agents.integrated_agent import IntegratedDiagnosticAgent


def demonstrate_explanation_features():
    """演示可解释AI功能"""
    print("=" * 80)
    print("可解释AI功能演示")
    print("=" * 80)
    
    # 创建代理实例
    agent = IntegratedDiagnosticAgent(tcm_database=None, wm_persist_dir=None)
    
    print("\n1. Chain-of-Thought (CoT) 推理演示")
    print("-" * 50)
    print("设置解释模式为 Chain-of-Thought...")
    result = agent.set_explanation_preference("/explain cot")
    print(result)
    
    print("\n2. 多粒度解释控制演示")
    print("-" * 50)
    print("a) 详细解释模式:")
    agent.set_explanation_preference("/explain detailed")
    print("   已切换到详细解释模式")
    
    print("\nb) 简洁解释模式:")
    agent.set_explanation_preference("/explain brief")
    print("   已切换到简洁解释模式")
    
    print("\nc) 结构化JSON输出模式:")
    agent.set_explanation_preference("/explain structured")
    print("   已切换到结构化JSON输出模式")
    
    print("\n3. 反事实解释演示")
    print("-" * 50)
    print("系统可以处理反事实查询，例如：")
    print("用户: 如果患者还有口干咽燥呢？")
    print("系统将分析改变特定症状后对诊断和治疗的影响")
    
    # 检测反事实查询
    cf_query = "如果患者还有口干咽燥呢？"
    is_cf = agent.explanation_component.is_counterfactual_query(cf_query)
    print(f"检测结果: '{cf_query}' -> 反事实查询: {is_cf}")
    
    print("\n4. 追问式解释演示")
    print("-" * 50)
    print("系统支持用户对解释本身提问，例如：")
    print("用户: 为什么不是心气虚？")
    print("系统将基于之前的诊断推理过程回答用户的问题")
    
    # 检测交互式查询
    interactive_query = "为什么不是心气虚？"
    context = "诊断: 脾气虚，治疗: 四君子汤"
    is_interactive = agent.explanation_component.is_interactive_query(interactive_query, context)
    print(f"检测结果: '{interactive_query}' -> 交互式查询: {is_interactive}")
    
    print("\n5. 依据溯源演示")
    print("-" * 50)
    print("系统可以要求模型引用知识来源，例如：")
    print("依据: '此辨证依据《中医诊断学》第5章：'舌淡主虚、主寒''")
    
    print("\n6. 对比解释演示")
    print("-" * 50)
    print("系统提供'为何选A不选B'的对比解释，例如：")
    print("推荐四君子汤而非补中益气汤，因患者无下陷症状（如脱肛、子宫下垂）")
    
    print("\n7. 结构化输出演示")
    print("-" * 50)
    print("JSON格式输出示例:")
    json_example = {
        "diagnosis": "脾气虚",
        "reasoning": [
            "主症：乏力、气短 → 气虚",
            "舌象：舌淡 → 虚证", 
            "无兼夹证 → 单纯脾气虚"
        ],
        "recommendation": "四君子汤加减",
        "confidence": 8,
        "reference": "《中医内科学》气虚证治"
    }
    print(f"示例JSON结构: {json_example}")
    
    print("\n8. 解释偏好切换演示")
    print("-" * 50)
    print("用户可以通过指令切换解释深度：")
    print("  /explain brief    → '因气虚，用补气药。'")
    print("  /explain detailed → 展开完整辨证逻辑链")
    print("  /explain cot      → Chain-of-Thought推理")
    print("  /explain structured → JSON结构化输出")
    
    print("\n" + "=" * 80)
    print("总结：通过API调用实现的可解释AI功能")
    print("=" * 80)
    print("✅ 已实现功能：")
    print("  - Chain-of-Thought推理")
    print("  - 反事实解释")
    print("  - 依据溯源")
    print("  - 模板化结构化输出")
    print("  - 多粒度解释控制")
    print("  - 追问式解释")
    print("  - 对比解释")
    print("  - 解释偏好设置")
    print("\n⚠️  API限制（无法实现）：")
    print("  - 可视化注意力热力图（无internal states输出）")
    print("  - 计算SHAP值（基于梯度）")
    print("  - 真实特征重要性排序")
    print("\n💡 核心思路：把模型当作'会自我解释的专家'，而不是'黑箱预测器'")


def demonstrate_api_based_xai():
    """演示基于API的XAI实现"""
    print("\n" + "=" * 80)
    print("基于API的XAI实现原理")
    print("=" * 80)
    
    principles = [
        "1. Chain-of-Thought (CoT) 推理",
        "   - 方法：在prompt中要求模型'先思考，再回答'",
        "   - 示例：'请逐步推理后给出答案'",
        "   - 优点：无需额外模块，兼容所有文本生成API"
    ],
    "2. 反事实解释",
    "   - 方法：主动询问'如果……会怎样？'",
    "   - 示例：'如果患者还有口干咽燥呢？'",
    "   - 实现：设计'解释追问'按钮，触发预设prompt",
    "",
    "3. 依据溯源",
    "   - 方法：要求模型引用知识来源（需在prompt中约束）",
    "   - 示例：'此辨证依据《中医诊断学》第5章'",
    "   - 注意：需结合外部知识库检索提升可靠性",
    "",
    "4. 模板化结构化输出",
    "   - 利用支持response_format={'type': 'json_object'}的API",
    "   - 优点：前端可折叠/展开解释，易于与系统集成",
    "",
    "5. 多粒度解释控制",
    "   - 通过用户指令切换解释深度",
    "   - 实现：在对话状态中记录'解释偏好'，动态调整prompt",
    "",
    "6. 追问式解释",
    "   - 设计Agent支持用户对解释本身提问",
    "   - 关键：维护决策上下文记忆，确保解释一致性",
    "",
    "7. 对比解释",
    "   - 主动提供'为何选A不选B'",
    "   - 示例：'推荐四君子汤而非补中益气汤，因患者无下陷症状'",
    "",
    "8. RAG + 解释增强",
    "   - 流程：检索知识库 → 结合用户输入 → 要求模型'基于以下资料解释'",
    "   - 效果：解释有据可依，减少幻觉"
    
    for principle in principles:
        print(principle)
    
    print("\n实践建议：")
    print("- 中医问诊Agent可解释性: CoT + RAG + JSON结构化输出")
    print("- 教学型Agent: 多粒度解释（学生模式 vs 专家模式）")
    print("- 多Agent系统: 专家Agent输出解释，学生Agent提问澄清")
    print("- 部署可行性: 全部基于API实现，无需本地模型")


def main():
    """主函数"""
    print("中西医结合诊疗系统 - 可解释AI功能演示")
    
    demonstrate_explanation_features()
    demonstrate_api_based_xai()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("该系统展示了如何通过API调用实现可解释AI功能")
    print("核心理念：通过'生成式解释'达到实用级可解释AI")
    print("=" * 80)


if __name__ == "__main__":
    main()