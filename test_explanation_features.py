#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试可解释AI功能
"""
import sys
import os
sys.path.append('/workspace')

from src.agents.integrated_agent import IntegratedDiagnosticAgent
from src.components.explanation_component import ExplanationComponent


def test_explanation_component():
    """测试解释组件功能"""
    print("=== 测试解释组件功能 ===")
    exp_component = ExplanationComponent()
    
    # 测试反事实查询检测
    print("\n1. 测试反事实查询检测:")
    cf_query = "如果患者还有口干咽燥呢？"
    is_cf = exp_component.is_counterfactual_query(cf_query)
    print(f"   查询: '{cf_query}'")
    print(f"   是否为反事实查询: {is_cf}")
    
    # 测试交互式查询检测
    print("\n2. 测试交互式查询检测:")
    interactive_query = "为什么选择这个方案？"
    context = "诊断: 脾气虚，方案: 四君子汤"
    is_interactive = exp_component.is_interactive_query(interactive_query, context)
    print(f"   查询: '{interactive_query}'")
    print(f"   上下文: '{context}'")
    print(f"   是否为交互式查询: {is_interactive}")
    
    # 测试生成对比解释
    print("\n3. 测试生成对比解释:")
    try:
        comparison_result = exp_component.generate_comparison_explanation(
            current_approach="四君子汤",
            alternative_approach="补中益气汤",
            patient_context="患者乏力、气短、舌淡，无下陷症状"
        )
        print(f"   当前方案: 四君子汤")
        print(f"   替代方案: 补中益气汤")
        print(f"   患者情况: 患者乏力、气短、舌淡，无下陷症状")
        print(f"   对比解释: {comparison_result[:200]}...")  # 只显示前200个字符
    except Exception as e:
        print(f"   生成对比解释时出现错误: {e}")
    
    # 测试生成反事实解释
    print("\n4. 测试生成反事实解释:")
    try:
        cf_result = exp_component.generate_counterfactual_explanation(
            original_symptoms="乏力、气短、舌淡",
            original_diagnosis="脾气虚",
            counterfactual_condition="如果患者还有口干咽燥"
        )
        print(f"   原始症状: 乏力、气短、舌淡")
        print(f"   原始诊断: 脾气虚")
        print(f"   反事实条件: 如果患者还有口干咽燥")
        print(f"   反事实解释: {cf_result[:200]}...")  # 只显示前200个字符
    except Exception as e:
        print(f"   生成反事实解释时出现错误: {e}")
    
    # 测试生成依据溯源解释
    print("\n5. 测试生成依据溯源解释:")
    try:
        citation_result = exp_component.generate_citation_explanation(
            diagnosis="脾气虚证",
            treatment="四君子汤加减"
        )
        print(f"   诊断: 脾气虚证")
        print(f"   治疗: 四君子汤加减")
        print(f"   依据溯源: {citation_result[:200]}...")  # 只显示前200个字符
    except Exception as e:
        print(f"   生成依据溯源解释时出现错误: {e}")
    
    # 测试生成交互式解释
    print("\n6. 测试生成交互式解释:")
    try:
        interactive_result = exp_component.generate_interactive_explanation(
            original_diagnosis="脾气虚证",
            reasoning_process="步骤1: 乏力、气短为气虚典型表现；步骤2: 舌淡进一步支持脾气虚；步骤3: 无热象或阴虚证候",
            user_question="为什么不是心气虚？"
        )
        print(f"   原始诊断: 脾气虚证")
        print(f"   推理过程: 步骤1: 乏力、气短为气虚典型表现；步骤2: 舌淡进一步支持脾气虚；步骤3: 无热象或阴虚证候")
        print(f"   用户问题: 为什么不是心气虚？")
        print(f"   交互式解释: {interactive_result[:200]}...")  # 只显示前200个字符
    except Exception as e:
        print(f"   生成交互式解释时出现错误: {e}")


def test_integrated_agent_explanation_features():
    """测试集成代理的可解释AI功能"""
    print("\n=== 测试集成代理可解释AI功能 ===")
    
    # 创建代理实例（不连接数据库以加快测试）
    agent = IntegratedDiagnosticAgent(tcm_database=None, wm_persist_dir=None)
    
    # 测试解释偏好设置
    print("\n1. 测试解释偏好设置:")
    result = agent.set_explanation_preference("/explain cot")
    print(f"   设置Chain-of-Thought模式: {result}")
    
    result = agent.set_explanation_preference("/explain detailed")
    print(f"   设置详细模式: {result}")
    
    result = agent.set_explanation_preference("/explain brief")
    print(f"   设置简洁模式: {result}")
    
    result = agent.set_explanation_preference("/explain structured")
    print(f"   设置结构化模式: {result}")
    
    # 测试提取诊断信息
    print("\n2. 测试提取诊断信息:")
    context = "诊断: 脾气虚\n分析: 乏力气短\n步骤1: 分析症状\n步骤2: 确定证型"
    prev_diag = agent._extract_previous_diagnosis(context)
    print(f"   上下文: {context}")
    print(f"   提取的诊断信息: {prev_diag}")
    
    reasoning = agent._extract_reasoning_process(context)
    print(f"   提取的推理过程: {reasoning}")
    
    # 测试JSON解析
    print("\n3. 测试JSON解析:")
    json_str = '{"diagnosis": "脾气虚", "reasoning": ["步骤1: 分析", "步骤2: 确定"], "recommendation": "四君子汤"}'
    parsed = agent._parse_json_result(json_str)
    print(f"   JSON字符串: {json_str}")
    print(f"   解析结果: {parsed}")
    
    formatted = agent._format_structured_output(parsed)
    print(f"   格式化输出: {formatted}")
    
    # 测试简化输出
    print("\n4. 测试简化输出:")
    long_text = "这是一个很长的文本。" * 50  # 重复50次
    simplified = agent._simplify_output(long_text)
    print(f"   原始文本长度: {len(long_text)}")
    print(f"   简化后长度: {len(simplified)}")
    print(f"   简化后内容: {simplified}")


def main():
    """主函数"""
    print("开始测试可解释AI功能...")
    
    test_explanation_component()
    test_integrated_agent_explanation_features()
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()