# 可解释AI (Explainable AI) 实现总结

## 概述

本项目基于API调用的大模型，实现了多种可解释AI功能，展示了如何通过"生成式解释"达到实用级可解释AI。核心理念是：**把模型当作"会自我解释的专家"，而不是"黑箱预测器"**。

## 已实现的可解释AI功能

### 1. 基础层级：事后解释 (Post-hoc Explanation)

#### 1.1 Chain-of-Thought (CoT) 推理
- **方法**：在 prompt 中要求模型"先思考，再回答"
- **实现**：通过 `cot_prompt` 模板和 `/explain cot` 指令
- **示例**：
  ```
  步骤1: [分析中医信息]
  步骤2: [分析西医信息] 
  步骤3: [对比分析]
  步骤4: [整合结论]
  ```

#### 1.2 反事实解释 (Counterfactual)
- **方法**：主动询问"如果……会怎样？"
- **实现**：`ExplanationComponent.is_counterfactual_query()` 和 `generate_counterfactual_explanation()`
- **示例**：用户问"如果患者还有口干咽燥呢？"，系统分析改变症状后对诊断的影响

#### 1.3 依据溯源 (Evidence Citation)
- **方法**：要求模型引用知识来源
- **实现**：`generate_citation_explanation()` 方法
- **示例**："'此辨证依据《中医诊断学》第5章：'舌淡主虚、主寒'"

### 2. 增强层级：结构化解释生成

#### 2.1 模板化输出 + JSON Schema
- **方法**：使用结构化JSON输出模板
- **实现**：`json_integration_prompt` 和 `/explain structured` 指令
- **输出格式**：
  ```json
  {
    "diagnosis": "诊断结论",
    "reasoning": ["推理步骤1", "推理步骤2", "推理步骤3"],
    "tcm_analysis": "中医分析",
    "wm_analysis": "西医分析", 
    "recommendation": "治疗建议",
    "confidence": "置信度(1-10)",
    "reference": "参考文献或依据"
  }
  ```

#### 2.2 多粒度解释控制
- **方法**：通过用户指令切换解释深度
- **实现**：
  - `/explain brief` → 简洁解释
  - `/explain detailed` → 详细解释
  - `/explain cot` → Chain-of-Thought推理
  - `/explain structured` → 结构化JSON输出

### 3. 高级层级：交互式可解释性

#### 3.1 追问式解释 (Interactive Q&A)
- **方法**：支持用户对解释本身提问
- **实现**：`is_interactive_query()` 和 `generate_interactive_explanation()`
- **示例**：用户问"为什么不是心气虚？"，系统基于推理过程回答

#### 3.2 对比解释 (Contrastive Explanation)
- **方法**：主动提供"为何选A不选B"
- **实现**：`generate_comparison_explanation()` 方法
- **示例**："推荐四君子汤而非补中益气汤，因患者无下陷症状"

## 技术实现细节

### 1. 新增组件
- `src/components/explanation_component.py`：解释组件，提供各种XAI功能
- 扩展 `src/agents/integrated_agent.py`：集成解释功能

### 2. 核心功能实现
- **Chain-of-Thought推理**：通过专门的提示模板实现逐步推理
- **反事实解释**：检测反事实查询并生成对比分析
- **交互式解释**：从上下文提取推理过程并回答用户问题
- **结构化输出**：使用JSON模板和解析器

### 3. 用户交互指令
- `/explain cot`：启用Chain-of-Thought推理模式
- `/explain detailed`：启用详细解释模式
- `/explain brief`：启用简洁解释模式
- `/explain structured`：启用结构化JSON输出模式

## API限制与解决方案

### 无法实现的能力
| 能力 | 原因 |
|------|------|
| 可视化注意力热力图 | 无 internal states 输出 |
| 计算 SHAP 值（基于梯度） | 无法获取梯度或 logits |
| 真实特征重要性排序 | 仅能通过输入扰动近似 |

### 已实现的替代方案
- **可解释性**：通过生成式解释实现
- **模型决策透明度**：通过Chain-of-Thought推理实现
- **决策依据**：通过引用和溯源实现

## 实践建议

| 需求 | 推荐方案 |
|------|--------|
| **中医问诊 Agent 可解释性** | CoT + RAG + JSON 结构化输出 |
| **教学型 Agent** | 多粒度解释（"学生模式" vs "专家模式"） |
| **多 Agent 系统** | 专家 Agent 输出解释，学生 Agent 提问澄清，形成解释闭环 |
| **部署可行性** | 全部基于 API 实现，无需本地模型 |

## 总结

通过API调用的模型，虽无内部透明性，但可通过"生成式解释"达到实用级可解释AI。本实现展示了：

1. **完整的XAI功能栈**：从基础到高级的多层级解释能力
2. **灵活的交互模式**：支持多种解释偏好和交互方式
3. **实用的部署方案**：完全基于API实现，无需本地模型
4. **可扩展的架构**：模块化设计，易于扩展新的解释功能

这种实现方式为AI系统提供了重要的可解释性，增强了用户信任和系统可靠性。