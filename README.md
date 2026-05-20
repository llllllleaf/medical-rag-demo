# 健康管理咨询师 GraphRAG 智能助手

> 基于 LangChain + LightRAG 的健康管理咨询师业务知识库,使用 GraphRAG（知识图谱增强检索）提升多跳问答准确率。

## 项目背景

在三胞医疗医院信息部任职期间,设计的健康管理咨询师辅助方案（POC 阶段获集团批准但未实施）的**可运行 demo 实现**。

**用户角色**：健康管理咨询师（非医生）
**用途**：辅助查询知识 + 培训陪练
**合规边界**：不涉及疾病诊断 / 不使用真实病历数据 / 不替代医生

## 解决的真实痛点

1. **查询效率低** — 健管师面对客户提问时查知识库慢
2. **新人培训成本高** — 新入职咨询师 1-3 个月才能熟悉业务
3. **回答标准不一** — 不同咨询师对同一问题回答不一致

## 技术栈

| 层 | 选型 | 理由 |
|----|------|------|
| RAG 框架 | LangChain + LightRAG | LangChain 入门 + LightRAG 实现 GraphRAG |
| LLM | 通义 qwen-turbo | 国内免 VPN / 中文优秀 / 有免费额度 |
| Embedding | 通义 text-embedding-v3 | 同账号省事 |
| 向量库 | Chroma | 本地零配置 |
| 前端 | Gradio | 10 行代码出 UI |
| 部署 | Vercel / HF Spaces | 一键部署 |

## 核心差异化

**普通 RAG vs GraphRAG**：
- 普通 RAG：文档切块 → 向量相似度检索 → 跨实体多跳推理会丢失关系
- GraphRAG：实体抽取 → 知识图谱 → 图遍历 + 向量,适合医学知识这种天然图谱

**示例对比问题**：
- "高血压客户咨询能不能空腹运动?" （多跳：高血压→服药→空腹→低血糖）
- "糖尿病客户问能不能喝下午茶咖啡?" （多跳：糖尿病→咖啡因→血糖影响）

## 快速启动

```bash
# 1. 克隆
git clone https://github.com/xxx/medical-rag-demo.git
cd medical-rag-demo

# 2. 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
export DASHSCOPE_API_KEY="你的通义 API Key"

# 5. 跑 Hello World
python hello.py

# 6. 跑 RAG 最小示例
python rag_minimal.py

# 7. 启动 Gradio UI
python app.py
```

## 开发进度

- [x] 项目初始化 + Python 环境
- [ ] LangChain Hello World
- [ ] 普通 RAG 最小闭环
- [ ] LightRAG Hello World
- [ ] 医疗数据集 + GraphRAG 集成
- [ ] Gradio UI + 对比展示
- [ ] 部署 + 博客 + 简历更新

## 配套博客

待发布于 [maoya.me](https://maoya.me)：
- 《为什么健管师 RAG 必须用 GraphRAG》
- 《从医院方案到可运行 demo 的实操路径》

## License

MIT
