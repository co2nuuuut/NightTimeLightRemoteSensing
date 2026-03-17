import networkx as nx
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import graphviz_layout

# 1. 定义集合 X 和邻接矩阵
nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
# 邻接矩阵 (0/1 表示关系是否存在)
#      a  b  c  d  e  f  g  h  i
adj_matrix = [
    [1, 0, 0, 0, 0, 1, 0, 0, 0], # a
    [0, 1, 1, 0, 0, 1, 0, 0, 0], # b
    [0, 0, 1, 0, 0, 0, 0, 0, 0], # c
    [0, 1, 1, 1, 0, 1, 0, 1, 1], # d
    [0, 0, 1, 0, 1, 1, 0, 0, 1], # e
    [0, 0, 0, 0, 0, 1, 0, 0, 0], # f
    [0, 0, 0, 0, 0, 1, 1, 0, 0], # g
    [0, 0, 1, 0, 0, 1, 0, 1, 1], # h
    [0, 0, 0, 0, 0, 0, 0, 0, 1], # i
]

# 2. 根据邻接矩阵创建一个有向图
# 我们只关心元素之间的关系，所以忽略自反边 (x, x)
G = nx.DiGraph()
for i, row in enumerate(adj_matrix):
    for j, val in enumerate(row):
        if i != j and val == 1:
            G.add_edge(nodes[i], nodes[j])

# 3. 计算传递规约 (Transitive Reduction) 来找到 Hasse 图的边
# 传递规约会移除图中的冗余边。例如，如果存在路径 a->b->c，
# 同时存在直接的边 a->c，那么 a->c 就是冗余的，将被移除。
# 这样剩下的边就代表了“覆盖”关系。
hasse_graph = nx.transitive_reduction(G)

# 4. 绘图
# 为了美观，我们可以手动指定节点的层级，以获得更清晰的 Hasse 图布局
# 这里我们根据观察将节点分层
pos = {
    'c': (1, 3), 'f': (3, 3), 'i': (5, 3),  # 顶层 (极大元)
    'b': (0, 2), 'h': (2, 2), 'e': (4, 2),  # 中间层
    'd': (1, 1), 'a': (2.5, 1), 'g': (3.5, 1), # 底层 (除了d之外都是极小元)
}

# 使用 graphviz_layout 也可以自动生成层次布局，但手动指定更可控
# pos = graphviz_layout(hasse_graph, prog="dot")


plt.figure(figsize=(8, 10)) # 设置画布大小

nx.draw(hasse_graph, pos,
        with_labels=True,
        node_size=2000,
        node_color='skyblue',
        font_size=16,
        font_weight='bold',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        width=2) # 边的宽度

plt.title("Hasse Diagram for (X, R)", size=20)
plt.show()

# 打印 Hasse 图的边（即覆盖关系）
print("Hasse 图的边 (覆盖关系):")
print(list(hasse_graph.edges()))