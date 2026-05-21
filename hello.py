"""
LangChain Hello World - 调用通义 qwen-turbo
跑通这一个文件 = 验证 LangChain + 通义 API 全链路通了
"""
from langchain_community.chat_models import ChatTongyi


def main():
    # 1. 创建 LLM 实例（不用手动传 API Key,会自动读环境变量 DASHSCOPE_API_KEY）
    llm = ChatTongyi(model="qwen-turbo")

    # 2. 提问
    question = "用 3 句话介绍一下 LangChain 是做什么的。"
    print(f"❓ 问题：{question}\n")

    # 3. 调用 LLM
    response = llm.invoke(question)

    # 4. 打印回答
    print(f"💬 通义回答：\n{response.content}")


if __name__ == "__main__":
    main()
