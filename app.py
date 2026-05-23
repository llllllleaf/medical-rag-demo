"""
Day 6 Final: Gradio UI - 健管师 RAG vs GraphRAG 对比 demo + 知识图谱可视化

【两个 Tab】
Tab 1: RAG 对比问答 (普通 RAG vs GraphRAG)
Tab 2: 知识图谱可视化 (140 实体 + 103 关系,可拖拽/缩放/点节点)

【启动】
python app.py → http://localhost:7860
"""
import asyncio
import json
import os

import gradio as gr
import networkx as nx
from pyvis.network import Network

# 普通 RAG
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# GraphRAG (LightRAG)
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


# ========== 配置 ==========
DATA_FILE = "data/health_consultant_kb.txt"
EVAL_FILE = "eval_questions.json"
CHROMA_DIR = "./chroma_db_v2"
LIGHTRAG_DIR = "./lightrag_storage_v2"
GRAPHML_FILE = f"{LIGHTRAG_DIR}/graph_chunk_entity_relation.graphml"
GRAPH_HTML = "graph_visualization.html"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = os.environ["DASHSCOPE_API_KEY"]

TOPIC_COLORS = {
    "高血压": "#e74c3c", "糖尿病": "#3498db", "失眠": "#9b59b6",
    "肥胖": "#f39c12", "营养": "#27ae60", "运动": "#16a085",
    "药": "#e67e22", "其他": "#95a5a6",
}


def classify_node(node_name: str) -> str:
    if any(k in node_name for k in ["高血压", "血压", "降压"]):
        return "高血压"
    if any(k in node_name for k in ["糖尿病", "血糖", "降糖", "胰岛素", "二甲双胍", "磺脲", "SGLT", "DPP"]):
        return "糖尿病"
    if any(k in node_name for k in ["失眠", "睡眠", "褪黑素"]):
        return "失眠"
    if any(k in node_name for k in ["肥胖", "BMI", "减重", "腰围"]):
        return "肥胖"
    if any(k in node_name for k in ["营养", "蛋白", "钠", "钾", "脂肪", "维生素", "燕麦", "深海鱼"]):
        return "营养"
    if any(k in node_name for k in ["运动", "有氧", "抗阻", "瑜伽", "快走", "慢跑"]):
        return "运动"
    if any(k in node_name for k in ["药", "ACEI", "ARB", "CCB", "卡托普利", "氯沙坦"]):
        return "药"
    return "其他"


# ========== 普通 RAG 初始化 ==========
print("🔵 初始化普通 RAG...")
loader = TextLoader(DATA_FILE, encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
embedding = DashScopeEmbeddings(model="text-embedding-v2")
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=CHROMA_DIR)
llm = ChatTongyi(model="qwen-turbo")
normal_qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)
print("   ✓ 普通 RAG 就绪")


# ========== GraphRAG 初始化 ==========
async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await openai_complete_if_cache(
        "qwen-turbo", prompt,
        system_prompt=system_prompt, history_messages=history_messages,
        api_key=API_KEY, base_url=DASHSCOPE_BASE_URL, **kwargs,
    )


async def emb_func(texts):
    return await openai_embed(
        texts, model="text-embedding-v2",
        api_key=API_KEY, base_url=DASHSCOPE_BASE_URL,
    )


async def init_graphrag():
    rag = LightRAG(
        working_dir=LIGHTRAG_DIR,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=emb_func),
    )
    await rag.initialize_storages()
    return rag


print("🟢 初始化 GraphRAG...")
graph_rag = asyncio.run(init_graphrag())
print("   ✓ GraphRAG 就绪")


# ========== 图谱信息 + 可视化预生成 ==========
g = nx.read_graphml(GRAPHML_FILE)
GRAPH_NODES = g.number_of_nodes()
GRAPH_EDGES = g.number_of_edges()


def generate_graph_html():
    """生成图谱可视化 HTML"""
    net = Network(
        height="700px", width="100%",
        bgcolor="#ffffff", font_color="#2c3e50",
        directed=True, notebook=False,
    )
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, spring_strength=0.04)
    for node in g.nodes():
        topic = classify_node(str(node))
        degree = g.degree(node)
        net.add_node(
            str(node), label=str(node),
            color=TOPIC_COLORS[topic], size=15 + degree * 3,
            title=f"<b>{node}</b><br>主题: {topic}<br>连接数: {degree}",
            group=topic,
        )
    for u, v, data in g.edges(data=True):
        desc = data.get("description", "")
        net.add_edge(str(u), str(v), title=desc[:200] if desc else "", value=float(data.get("weight", 1.0)))
    net.write_html(GRAPH_HTML, notebook=False, open_browser=False)
    return GRAPH_HTML


print("🎨 生成知识图谱可视化...")
generate_graph_html()
print(f"   ✓ 已生成 {GRAPH_HTML}")

# 用 iframe + Gradio 静态文件服务(srcdoc 嵌大文件会导致前端崩溃)
GRAPH_IFRAME_HTML = (
    f'<iframe src="/gradio_api/file={GRAPH_HTML}" '
    f'width="100%" height="750" '
    f'style="border:1px solid #e0e0e0; border-radius:8px;"></iframe>'
)


# ========== 加载示例问题 ==========
with open(EVAL_FILE, "r", encoding="utf-8") as f:
    eval_data = json.load(f)
EXAMPLE_QUESTIONS = [q["question"] for q in eval_data["questions"]]


# ========== 查询函数 ==========
def ask_normal_rag(question: str) -> str:
    if not question.strip():
        return ""
    try:
        result = normal_qa.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"❌ 报错: {e}"


async def ask_graph_rag_async(question: str) -> str:
    if not question.strip():
        return ""
    try:
        return await graph_rag.aquery(question, param=QueryParam(mode="hybrid"))
    except Exception as e:
        return f"❌ 报错: {e}"


async def compare(question: str):
    """Gradio 6.0 支持 async callback,直接 await 不需要自建 event loop"""
    if not question.strip():
        return "请输入问题", "请输入问题"
    # 普通 RAG 是同步的,直接调
    normal_answer = ask_normal_rag(question)
    # GraphRAG 是异步的,await
    graph_answer = await ask_graph_rag_async(question)
    return normal_answer, graph_answer


# ========== Gradio UI ==========
with gr.Blocks(
    title="健管师 GraphRAG 智能助手",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        f"""
        # 🏥 健康管理咨询师 GraphRAG 智能助手

        > **数据集**: 6 主题健管师 KB(高血压/糖尿病/失眠/肥胖/营养/运动,3209 字)
        > **知识图谱**: 实体数 **{GRAPH_NODES}** | 关系数 **{GRAPH_EDGES}**
        > **技术栈**: LangChain + LightRAG + 通义 qwen-turbo + Chroma + Gradio
        """
    )

    with gr.Tabs():
        # ========== Tab 1: RAG 对比问答 ==========
        with gr.Tab("💬 RAG 对比问答"):
            gr.Markdown("**使用方式**:点示例问题或自行输入 → 两个 RAG 同时回答 → 对比差异")

            with gr.Row():
                with gr.Column(scale=4):
                    question_input = gr.Textbox(
                        label="💬 问题",
                        placeholder="例如:糖尿病客户能不能空腹晨练?",
                        lines=2,
                    )
                with gr.Column(scale=1):
                    ask_btn = gr.Button("🚀 提问", variant="primary", size="lg")

            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question_input,
                label="🎯 10 个示例问题(点击直接填充)",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔵 普通 RAG (Chroma + top-k=3)")
                    normal_output = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### 🟢 GraphRAG (LightRAG hybrid)")
                    graph_output = gr.Markdown()

            ask_btn.click(fn=compare, inputs=question_input, outputs=[normal_output, graph_output])

        # ========== Tab 2: 知识图谱可视化 ==========
        with gr.Tab("🕸️ 知识图谱可视化"):
            gr.Markdown(
                f"""
                ### 由 LightRAG 自动从文档抽取构建

                **{GRAPH_NODES} 个实体** + **{GRAPH_EDGES} 条关系**
                颜色编码: 🔴 高血压 | 🔵 糖尿病 | 🟣 失眠 | 🟠 肥胖 | 🟢 营养 | 🟦 运动 | 🟧 药 | ⚪ 其他

                **操作**: 拖拽节点 / 滚轮缩放 / 悬停看详情 / 底部物理面板调引力
                """
            )
            gr.HTML(GRAPH_IFRAME_HTML)

        # ========== Tab 3: 项目说明 ==========
        with gr.Tab("📖 项目说明"):
            gr.Markdown(
                f"""
                ## 项目背景

                在三胞医疗医院信息部任职期间设计的"健康管理咨询师辅助方案" POC 阶段获集团批准但未实施,
                现把它做成可运行 demo,补足简历"方案规划→落地"的差距。

                ## 解决的痛点

                1. **查询效率低** — 健管师查公司知识库慢
                2. **新人培训成本高** — 1-3 个月才熟悉业务
                3. **回答标准不一** — 不同咨询师对同一问题回答不一致

                ## 技术栈

                | 层 | 选型 | 理由 |
                |----|------|------|
                | RAG 框架 | LangChain + LightRAG | LightRAG 实现 GraphRAG,中文友好 |
                | LLM | 通义 qwen-turbo | 国内免 VPN + 中文优秀 |
                | Embedding | 通义 text-embedding-v2 | 1536 维,与 LightRAG 默认匹配 |
                | 向量库 | Chroma | 本地零配置 |
                | 前端 | Gradio | 10 行出 UI |

                ## 核心差异化:普通 RAG vs GraphRAG

                - **普通 RAG**: 文档切块 → 向量相似度 → top-k 检索 → LLM 回答
                - **GraphRAG**: 同样切块 + **LLM 抽取实体/关系自动建知识图谱** → 查询时图遍历 + 向量综合

                **真正的智能不在图数据库,在 LLM 的实体识别能力** —— 图数据库只是存储容器。

                ## 合规边界

                - ✅ 服务对象: 健康管理咨询师(非医生)
                - ✅ 用途: 辅助查询 + 培训陪练
                - ❌ 不涉及: 疾病诊断 / 处方推荐
                - ✅ 数据: 公开健康科普 + 自制 FAQ,**不使用任何真实病历**

                ## 项目链接

                - GitHub: <https://github.com/llllllleaf/medical-rag-demo>
                - 完整对比报告: comparison.md (GitHub 仓库内,433 行)
                """
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        allowed_paths=[GRAPH_HTML],  # 允许 iframe 访问这个静态文件
    )
