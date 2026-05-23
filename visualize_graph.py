"""
知识图谱可视化 - 用 pyvis 把 LightRAG 的 graphml 渲染成可交互 HTML

【输出】graph_visualization.html
浏览器打开后可以:
  - 拖拽节点
  - 缩放
  - 点节点查看详情
  - 按主题查看不同颜色
"""
import networkx as nx
from pyvis.network import Network

GRAPH_FILE = "lightrag_storage_v2/graph_chunk_entity_relation.graphml"
OUTPUT_FILE = "graph_visualization.html"


# 6 大主题 → 颜色映射(基于实体名关键词分类)
TOPIC_COLORS = {
    "高血压": "#e74c3c",  # 红
    "糖尿病": "#3498db",  # 蓝
    "失眠": "#9b59b6",    # 紫
    "肥胖": "#f39c12",    # 橙
    "营养": "#27ae60",    # 绿
    "运动": "#16a085",    # 青绿
    "药": "#e67e22",      # 暗橙(药物相关)
    "其他": "#95a5a6",    # 灰
}


def classify_node(node_name: str) -> str:
    """根据节点名归类主题"""
    name = node_name.lower()
    # 优先级:疾病 > 药物 > 营养/运动 > 其他
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
    if any(k in node_name for k in ["运动", "有氧", "抗阻", "瑜伽", "运动强度", "快走", "慢跑"]):
        return "运动"
    if any(k in node_name for k in ["药", "ACEI", "ARB", "CCB", "卡托普利", "氯沙坦"]):
        return "药"
    return "其他"


def main():
    print(f"📂 读取图谱: {GRAPH_FILE}")
    g = nx.read_graphml(GRAPH_FILE)
    print(f"   ✓ {g.number_of_nodes()} 节点 / {g.number_of_edges()} 边")

    # 创建 pyvis 网络
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2c3e50",
        directed=True,
        notebook=False,
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
    )

    # 添加节点
    topic_counts = {}
    for node in g.nodes():
        topic = classify_node(str(node))
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        color = TOPIC_COLORS[topic]
        # 节点的度数(连接多少条边)决定节点大小
        degree = g.degree(node)
        size = 15 + degree * 3
        # 节点详情(鼠标悬停显示)
        title = f"<b>{node}</b><br>主题: {topic}<br>连接数: {degree}"
        net.add_node(
            str(node),
            label=str(node),
            color=color,
            size=size,
            title=title,
            group=topic,
        )

    # 添加边
    for u, v, data in g.edges(data=True):
        desc = data.get("description", "")
        weight = data.get("weight", 1.0)
        title = desc[:200] if desc else ""
        net.add_edge(str(u), str(v), title=title, value=float(weight))

    # 配置选项(交互菜单)
    net.show_buttons(filter_=["physics"])

    # 写入 HTML
    net.write_html(OUTPUT_FILE, notebook=False, open_browser=False)
    print(f"\n📊 主题分布:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"   {topic}: {count} 个节点")
    print(f"\n✓ 可视化已生成: {OUTPUT_FILE}")
    print(f"  浏览器打开看效果")


if __name__ == "__main__":
    main()
