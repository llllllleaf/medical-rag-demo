"""
LightRAG Hello World - 用通义 qwen-turbo 跑 GraphRAG

跑通这个文件 = 你能用 LightRAG 构建知识图谱并查询
"""
import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# ========== 配置 ==========
WORKING_DIR = "./lightrag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

# 通义 OpenAI 兼容接口(LightRAG 没内置通义,但通义有 OpenAI 兼容模式)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ["DASHSCOPE_API_KEY"]


# ========== LLM 调用函数(异步)==========
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """LightRAG 会用这个函数做实体抽取和回答生成"""
    return await openai_complete_if_cache(
        "qwen-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        **kwargs,
    )


# ========== Embedding 函数(异步)==========
# 用通义 text-embedding-v2(1536 维,跟 LightRAG 默认一致,省去维度配置麻烦)
async def embedding_func(texts):
    """LightRAG 会用这个函数把文本转向量"""
    return await openai_embed(
        texts,
        model="text-embedding-v2",
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


# ========== 初始化 LightRAG ==========
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,        # text-embedding-v2 的维度
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    await rag.initialize_storages()
    return rag


# ========== 主流程 ==========
async def main():
    print("=" * 60)
    print("🚀 LightRAG Hello World - GraphRAG 演示")
    print("=" * 60)

    print("\n📦 [1/3] 初始化 LightRAG...")
    rag = await initialize_rag()
    print("   ✓ 已就绪")

    print("\n📄 [2/3] 插入 test.txt + 自动构建知识图谱（这步会调 LLM 抽取实体,5-10 分钟）...")
    with open("test.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())
    print(f"   ✓ 知识图谱已存到 {WORKING_DIR}/")
    print(f"   ✓ 看看 {WORKING_DIR}/graph_chunk_entity_relation.graphml 就是知识图谱文件")

    print("\n🤖 [3/3] 用 GraphRAG 查询(hybrid 模式 = 图遍历 + 向量)...")
    question = "高血压客户能不能空腹运动?"
    print(f"\n❓ 问题: {question}")

    result = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    print(f"\n💬 GraphRAG 回答:\n{result}")


if __name__ == "__main__":
    asyncio.run(main())
