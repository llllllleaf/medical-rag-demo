"""
RAG 最小闭环 - 5 步示例
用 test.txt（高血压健管师知识库）演示完整 RAG 流程
"""
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQA


def main():
    print("=" * 60)
    print("🚀 RAG 最小闭环 - 5 步演示")
    print("=" * 60)

    # ---------- 第 1 步:加载文档 ----------
    print("\n📄 [1/5] 加载 test.txt ...")
    loader = TextLoader("test.txt", encoding="utf-8")
    docs = loader.load()
    print(f"   ✓ 加载完成,共 {len(docs)} 个文档,总字数 {len(docs[0].page_content)}")

    # ---------- 第 2 步:切分文档 ----------
    print("\n✂️  [2/5] 切分文档（chunk_size=500, overlap=50）...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每块 500 字
        chunk_overlap=50,    # 块之间重叠 50 字（避免边界信息丢失）
    )
    chunks = splitter.split_documents(docs)
    print(f"   ✓ 切分完成,共 {len(chunks)} 个块")

    # ---------- 第 3 步:Embedding（把文字变向量）----------
    print("\n🔢 [3/5] 用通义 Embedding 转向量 ...")
    embedding = DashScopeEmbeddings(model="text-embedding-v3")

    # ---------- 第 4 步:存到向量库 Chroma ----------
    print("\n💾 [4/5] 存入 Chroma 向量库 ...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db",  # 持久化目录(下次跑不用重新构建)
    )
    print(f"   ✓ {len(chunks)} 个块已存入向量库")

    # ---------- 第 5 步:检索 + 生成 ----------
    print("\n🤖 [5/5] 构建 RAG 链(检索 + LLM 回答)...")
    llm = ChatTongyi(model="qwen-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),  # 检索 top 3 相关块
        return_source_documents=True,  # 返回引用的源文档(可解释性)
    )

    # ---------- 提问 ----------
    questions = [
        "高血压客户咨询能不能空腹运动?",
        "高血压客户能喝咖啡吗?",
        "高血压的分级标准是什么?",
    ]

    for i, question in enumerate(questions, 1):
        print("\n" + "=" * 60)
        print(f"❓ 问题 {i}: {question}")
        print("=" * 60)
        result = qa.invoke({"query": question})
        print(f"\n💬 回答:\n{result['result']}")
        print(f"\n📚 引用了 {len(result['source_documents'])} 个文档块")


if __name__ == "__main__":
    main()
