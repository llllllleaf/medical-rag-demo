"""
Day 5.4: 对比测试 普通 RAG vs GraphRAG

【做什么】
1. 读 eval_questions.json 的 10 个问题
2. 对每个问题,普通 RAG (Chroma 向量库) 和 GraphRAG (LightRAG) 各答一次
3. 输出 comparison.md 对比表

【面试核心证据】
"我用同样的数据 + 同样的 LLM,跑了 10 个问题,普通 RAG 在 6 个多跳问题上明显答不全,
GraphRAG 凭借知识图谱能跨主题推理,这就是我选择 GraphRAG 的数据依据。"
"""
import asyncio
import json
import os
from datetime import datetime

# ---- 普通 RAG ----
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQA

# ---- GraphRAG (LightRAG) ----
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ========== 配置 ==========
DATA_FILE = "data/health_consultant_kb.txt"
EVAL_FILE = "eval_questions.json"
OUTPUT_FILE = "comparison.md"

CHROMA_DIR = "./chroma_db_v2"      # 普通 RAG 的向量库(用新数据集建)
LIGHTRAG_DIR = "./lightrag_storage_v2"  # GraphRAG 的存储(已建好)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ["DASHSCOPE_API_KEY"]


# ========== 普通 RAG 初始化 ==========
def init_normal_rag():
    """基于新数据建普通 RAG (Chroma 向量库)"""
    print("📦 初始化普通 RAG...")
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"   ✓ 切分: {len(chunks)} 个块")

    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
    )
    print(f"   ✓ 向量库已存到 {CHROMA_DIR}/")

    llm = ChatTongyi(model="qwen-turbo")
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    return qa


# ========== GraphRAG 初始化 ==========
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        "qwen-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts):
    return await openai_embed(
        texts,
        model="text-embedding-v2",
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


async def init_graphrag():
    """用已构建好的 lightrag_storage_v2"""
    print("📦 初始化 GraphRAG (LightRAG)...")
    rag = LightRAG(
        working_dir=LIGHTRAG_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    print(f"   ✓ 加载已有图谱 {LIGHTRAG_DIR}/")
    return rag


# ========== 主对比流程 ==========
async def main():
    print("=" * 70)
    print("🚀 Day 5.4 - 对比测试 普通 RAG vs GraphRAG")
    print("=" * 70)

    # 初始化两个 RAG
    normal_qa = init_normal_rag()
    graph_rag = await init_graphrag()

    # 读问题
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    questions = eval_data["questions"]
    print(f"\n📋 共 {len(questions)} 个问题待测试\n")

    results = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        qtype = q["type"]
        topics = ", ".join(q["cross_topics"])

        print(f"{'='*70}")
        print(f"❓ [{i}/{len(questions)}] Q{qid}: {question}")
        print(f"   类型: {qtype} | 跨主题: {topics}")

        # 普通 RAG 答
        print(f"\n   🔵 普通 RAG 回答中...")
        try:
            normal_result = normal_qa.invoke({"query": question})
            normal_answer = normal_result["result"]
            normal_chunks = len(normal_result.get("source_documents", []))
        except Exception as e:
            normal_answer = f"[报错] {e}"
            normal_chunks = 0

        # GraphRAG 答
        print(f"   🟢 GraphRAG 回答中...")
        try:
            graph_answer = await graph_rag.aquery(
                question, param=QueryParam(mode="hybrid")
            )
        except Exception as e:
            graph_answer = f"[报错] {e}"

        results.append({
            "id": qid,
            "question": question,
            "type": qtype,
            "cross_topics": topics,
            "normal_answer": normal_answer,
            "normal_chunks": normal_chunks,
            "graph_answer": graph_answer,
        })

        # 中间预览(前 80 字)
        print(f"\n   🔵 普通 RAG: {str(normal_answer)[:80]}...")
        print(f"   🟢 GraphRAG: {str(graph_answer)[:80]}...")

    # ========== 输出 comparison.md ==========
    print(f"\n{'='*70}")
    print(f"📝 写入对比报告 {OUTPUT_FILE}...")

    md = []
    md.append("# 普通 RAG vs GraphRAG 对比测试报告\n")
    md.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"> 数据集: `{DATA_FILE}` (6 主题 / 3209 字)\n")
    md.append(f"> 共 {len(questions)} 个测试问题\n\n")

    md.append("## 测试配置\n\n")
    md.append("| 项 | 普通 RAG | GraphRAG (LightRAG) |\n")
    md.append("|----|---------|---------------------|\n")
    md.append("| LLM | 通义 qwen-turbo | 通义 qwen-turbo |\n")
    md.append("| Embedding | text-embedding-v2 | text-embedding-v2 |\n")
    md.append("| 向量库 | Chroma | NanoVectorDB (LightRAG 内置) |\n")
    md.append("| 检索方式 | top-k=3 向量相似度 | hybrid (local + global + 向量) |\n")
    md.append("| 知识图谱 | ❌ 无 | ✅ 140 节点 + 103 边 |\n\n")

    md.append("---\n\n")
    md.append("## 详细对比\n\n")

    for r in results:
        md.append(f"### Q{r['id']}: {r['question']}\n\n")
        md.append(f"- **类型**: `{r['type']}` | **跨主题**: {r['cross_topics']}\n\n")

        md.append(f"#### 🔵 普通 RAG\n\n")
        md.append(f"> 引用 {r['normal_chunks']} 个 chunk\n\n")
        md.append(f"{r['normal_answer']}\n\n")

        md.append(f"#### 🟢 GraphRAG (LightRAG hybrid)\n\n")
        md.append(f"{r['graph_answer']}\n\n")
        md.append("---\n\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"   ✓ 报告已生成: {OUTPUT_FILE}")
    print(f"\n🎉 Day 5.4 完成!")


if __name__ == "__main__":
    asyncio.run(main())
