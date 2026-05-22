"""
LightRAG Hello World - 用通义 qwen-turbo 跑 GraphRAG

【这个文件做什么】
1. 把 test.txt 喂给 LightRAG
2. LightRAG 自动从文档抽取"实体 + 关系"构建知识图谱
3. 提问 → 沿知识图谱遍历 + 向量检索 → LLM 综合回答

【面试官可能会问】
Q: 你这个文件为什么用 asyncio?
A: LightRAG 内部是异步设计(I/O 密集,LLM 调用、Embedding 都是网络请求),用 async 性能高 10x

Q: 你为什么不用 LightRAG 自带的 OpenAI 后端,要自己写 llm_model_func?
A: LightRAG 1.4.16 没内置通义后端,但通义有 OpenAI 兼容接口(dashscope.aliyuncs.com/compatible-mode/v1),
   所以我用 openai_complete_if_cache 函数 + 通义的兼容地址实现"伪 OpenAI 调用"
"""

# ========== 导入依赖 ==========
import asyncio  # Python 内置:异步编程框架
import os       # Python 内置:读环境变量

# LightRAG 核心类(rag = 实例;QueryParam = 查询配置)
from lightrag import LightRAG, QueryParam

# LightRAG 提供的 OpenAI 兼容调用工具
# openai_complete_if_cache: LLM 调用 + 内置缓存(同一问题第二次问不再调 LLM)
# openai_embed: Embedding(文本转向量)
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

# EmbeddingFunc: 把"embedding 函数 + 维度信息"包装成 LightRAG 能识别的对象
from lightrag.utils import EmbeddingFunc


# ========== 全局配置 ==========

# LightRAG 把知识图谱 + 向量库 + 元数据存这个目录
# 跑完后这里会有:graph_chunk_entity_relation.graphml (图谱) + 多个 vdb_*.json (向量库) + kv_store_*.json (缓存)
WORKING_DIR = "./lightrag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)  # 不存在就创建,存在不报错

# 通义的 OpenAI 兼容接口地址(关键!这是用 OpenAI SDK 调通义的诀窍)
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 从环境变量读 API Key(避免硬编码,防止误提交 GitHub)
API_KEY = os.environ["DASHSCOPE_API_KEY"]


# ========== LLM 调用函数(异步) ==========
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    """
    LightRAG 在 2 个场景会调这个函数:
      1. 插入文档时:用 LLM 从文本抽取"实体"和"关系"
      2. 查询时:把检索到的内容 + 问题塞给 LLM,生成最终回答

    参数:
      prompt: 给 LLM 的实际问题
      system_prompt: 系统提示词(可选)
      history_messages: 多轮对话历史(可选,本 demo 没用)
      **kwargs: LightRAG 可能传的其他参数(如 temperature)

    返回: LLM 的字符串回答
    """
    return await openai_complete_if_cache(
        "qwen-turbo",              # 通义入门款,便宜且够用
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        **kwargs,
    )


# ========== Embedding 函数(异步) ==========
# 用通义 text-embedding-v2(1536 维,跟 LightRAG 默认一致,省去维度配置麻烦)
# 之前试过 v3(1024 维)但 LightRAG 默认 1536 维,会报维度不匹配
async def embedding_func(texts):
    """
    把 List[str] 文本数组 → 转成 List[List[float]] 向量数组
    LightRAG 在 2 个场景调:
      1. 插入文档时:每个 chunk 转向量存到向量库
      2. 查询时:把问题转向量,从向量库找最相似的 chunk
    """
    return await openai_embed(
        texts,
        model="text-embedding-v2",    # 通义 embedding v2,1536 维
        api_key=API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


# ========== 初始化 LightRAG ==========
async def initialize_rag():
    """
    创建 LightRAG 实例并初始化存储(向量库、KV 库、图谱库)
    必须在所有 insert/query 之前调用一次
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,            # 存储目录
        llm_model_func=llm_model_func,      # 上面定义的 LLM 调用函数
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,             # text-embedding-v2 的维度(关键!不一致会报错)
            max_token_size=8192,            # 单次最多处理 8192 token
            func=embedding_func,            # 上面定义的 embedding 函数
        ),
    )
    # 初始化 3 个底层存储:
    # 1. KV 存储(JsonKVStorage):存 LLM 调用缓存、文档元数据
    # 2. 向量存储(NanoVectorDBStorage):存 chunk/entity/relation 的 embedding
    # 3. 图存储(NetworkXStorage):存知识图谱(实体 + 关系)
    await rag.initialize_storages()
    return rag


# ========== 主流程 ==========
async def main():
    print("=" * 60)
    print("🚀 LightRAG Hello World - GraphRAG 演示")
    print("=" * 60)

    # ---------- Step 1:初始化 ----------
    print("\n📦 [1/3] 初始化 LightRAG...")
    rag = await initialize_rag()
    print("   ✓ 已就绪")

    # ---------- Step 2:插入文档(自动构建知识图谱) ----------
    # 这一步是 GraphRAG 跟普通 RAG 的核心差异:
    #   普通 RAG: 切块 → 转向量 → 存(完了)
    #   GraphRAG: 切块 → 转向量 → 存 + 调 LLM 抽取实体和关系 → 构建图谱
    # 所以这一步会调多次 LLM,5-10 分钟很正常
    print("\n📄 [2/3] 插入 test.txt + 自动构建知识图谱(这步会调 LLM 抽取实体,5-10 分钟)...")
    with open("test.txt", "r", encoding="utf-8") as f:
        # ainsert = async insert,异步插入
        # LightRAG 内部会:1)切块 2)用 embedding_func 转向量 3)用 llm_model_func 抽实体关系 4)存到 working_dir
        await rag.ainsert(f.read())
    print(f"   ✓ 知识图谱已存到 {WORKING_DIR}/")
    print(f"   ✓ 看看 {WORKING_DIR}/graph_chunk_entity_relation.graphml 就是知识图谱文件")

    # ---------- Step 3:查询 ----------
    print("\n🤖 [3/3] 用 GraphRAG 查询(hybrid 模式 = 图遍历 + 向量)...")
    question = "高血压客户能不能空腹运动?"
    print(f"\n❓ 问题: {question}")

    # aquery = async query,异步查询
    # mode 取值:
    #   "naive"  = 只用向量检索(=普通 RAG,不用图谱)
    #   "local"  = 只用图谱的"局部实体"检索(图谱遍历)
    #   "global" = 用图谱的"全局关系"检索(高层抽象)
    #   "hybrid" = local + global + 向量 综合(推荐,准确率最高)
    #   "mix"    = LightRAG 1.4+ 新增,跟 hybrid 类似但更激进
    result = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    print(f"\n💬 GraphRAG 回答:\n{result}")


# Python 入口:if __name__ == "__main__" 表示"直接运行这个文件时执行"
# asyncio.run(main()) = 启动异步事件循环跑 main()
if __name__ == "__main__":
    asyncio.run(main())
