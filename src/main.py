"""
中西医结合诊疗系统 - 主入口
功能：
1. 智能回答中西医相关问题
2. 支持逐步问诊功能
3. 整合中西医诊断信息
4. 记忆对话历史和患者信息
5. 提供可解释AI功能（Chain-of-Thought推理、反事实解释、追问式解释等）
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.agents.integrated_agent import IntegratedDiagnosticAgent


def main():
    """主函数 - 交互式界面"""
    agent = IntegratedDiagnosticAgent()
    
    print("=" * 60)
    print("中西医结合诊疗系统 - Final Agent (支持可解释AI)")
    print("=" * 60)
    print("功能说明：")
    print("1. 智能回答中西医相关问题")
    print("2. 支持逐步问诊功能")
    print("3. 整合中西医诊断信息")
    print("4. 记忆对话历史和患者信息")
    print("5. 提供可解释AI功能：")
    print("   - Chain-of-Thought推理: /explain cot")
    print("   - 详细解释: /explain detailed")
    print("   - 简洁解释: /explain brief")
    print("   - 结构化输出: /explain structured")
    print("   - 反事实解释: '如果...会怎样？'")
    print("   - 追问式解释: '为什么...?' '如何...?'")
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