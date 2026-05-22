"""
Day 5.2: 用扩展后的健管师 KB(6 主题 / 3200 字)重建 GraphRAG 知识图谱

【做什么】
1. 读 data/health_consultant_kb.txt
2. 用 LightRAG 抽取实体 + 关系,构建知识图谱
3. 存到 lightrag_storage_v2/ (跟旧的高血压图谱分开)

【面试可讲】
"我把数据从单主题(高血压)扩展到 6 个主题,LightRAG 会自动建立跨主题的实体关系图谱,
这正是 GraphRAG 比普通 RAG 强的关键 —— 跨主题问题需要多跳推理。"
"""
import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


WORKING_DIR = "./lightrag_storage_v2"  # 新目录,不影响旧 lightrag_storage
os.makedirs(WORKING_DIR, exist_ok=True)

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ["DASHSCOPE_API_KEY"]
DATA_FILE = "data/health_consultant_kb.txt"


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


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag


async def main():
    print("=" * 60)
    print("🚀 Day 5.2 - 用 6 主题 KB 重建 GraphRAG 知识图谱")
    print("=" * 60)

    print(f"\n📦 [1/3] 初始化 LightRAG (working_dir={WORKING_DIR})...")
    rag = await initialize_rag()
    print("   ✓ 已就绪")

    print(f"\n📄 [2/3] 加载 {DATA_FILE} ...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    print(f"   ✓ 数据加载: {len(content)} 字符")

    print(f"\n🔨 [3/3] 插入 + 构建知识图谱(10-20 分钟,调多次 LLM 抽实体关系)...")
    await rag.ainsert(content)
    print(f"   ✓ 知识图谱构建完成")

    # 统计图谱规模
    print("\n📊 知识图谱统计:")
    print(f"   存储目录: {WORKING_DIR}/")
    import os as _os
    for f in sorted(_os.listdir(WORKING_DIR)):
        size = _os.path.getsize(f"{WORKING_DIR}/{f}")
        print(f"   - {f} ({size:,} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
