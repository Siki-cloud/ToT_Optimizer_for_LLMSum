# src/search_visualizer.py
import json
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Any, Optional

def visualize_search_tree(tree_data: Dict[str, Any], output_file: str = "search_tree.html") -> Optional[go.Figure]:
    """
    可视化搜索树 - 使用最新的plotly API
    
    Args:
        tree_data: 搜索树数据
        output_file: 输出文件路径
        
    Returns:
        plotly图形对象或None
    """
    try:
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node_info in tree_data["nodes"].items():
            state = node_info.get("state", [])
            label = " → ".join(state) if state else "根"
            score = node_info.get("score", 0.0)
            
            G.add_node(node_id, 
                      label=label,
                      score=score,
                      state=state,
                      depth=node_info.get("depth", 0),
                      visits=node_info.get("visits", 0))
        
        # 添加边
        for edge in tree_data["edges"]:
            G.add_edge(edge["from"], edge["to"])
        
        # 计算布局
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # 提取节点数据
        node_x, node_y = [], []
        node_text, node_color, node_size = [], [], []
        node_labels = []
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node_id]
            label = node_info["label"]
            score = node_info["score"]
            visits = node_info["visits"]
            
            # 悬停文本
            hover_text = f"{label}<br>分数: {score:.3f}<br>访问次数: {visits}"
            node_text.append(hover_text)
            node_labels.append(label)
            
            # 颜色和大小
            node_color.append(score)
            size = 15 + visits * 1.5
            node_size.append(min(size, 50))
        
        # 创建边轨迹
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color='rgba(136, 136, 136, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 创建节点轨迹 - 使用正确的colorbar配置
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text='节点分数',
                        side='right'
                    ),
                    xanchor='left',
                    len=0.5
                ),
                line=dict(width=2, color='DarkSlateGrey')
            )
        )
        
        # 创建布局 - 使用正确的title配置
        layout = go.Layout(
            title=dict(
                text='🌳 ToT搜索树可视化',
                font=dict(size=18, family="Arial"),
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                visible=False
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                visible=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # 创建图形
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        
        # 保存为HTML文件
        fig.write_html(output_file)
        print(f"✅ 搜索树可视化已保存到: {output_file}")
        
        return fig
        
    except ImportError as e:
        print(f"⚠️  缺少必要的库: {e}")
        print("请安装: pip install plotly networkx")
        return None
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return None


def visualize_search_tree_simple(tree_data: Dict[str, Any], output_file: str = "search_tree_simple.html") -> Optional[go.Figure]:
    """
    简化的搜索树可视化 - 避免使用可能过时的配置
    
    Args:
        tree_data: 搜索树数据
        output_file: 输出文件路径
        
    Returns:
        plotly图形对象或None
    """
    try:
        import plotly.graph_objects as go
        import networkx as nx
        
        # 创建图
        G = nx.DiGraph()
        
        # 添加节点
        for node_id, node_info in tree_data["nodes"].items():
            state = node_info.get("state", [])
            label = " → ".join(state) if state else "根"
            G.add_node(node_id, 
                      label=label,
                      score=node_info.get("score", 0.0),
                      visits=node_info.get("visits", 0))
        
        # 添加边
        for edge in tree_data["edges"]:
            G.add_edge(edge["from"], edge["to"])
        
        # 布局
        pos = nx.spring_layout(G, seed=42)
        
        # 边轨迹
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # 节点轨迹
        node_x, node_y = [], []
        node_text, node_color, node_size = [], [], []
        
        for node_id in G.nodes():
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node_id]
            hover_text = f"{node_info['label']}<br>分数: {node_info['score']:.3f}<br>访问: {node_info['visits']}"
            node_text.append(hover_text)
            node_color.append(node_info['score'])
            node_size.append(15 + min(node_info['visits'] * 1.5, 30))
        
        # 简化的节点配置
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            text=node_text,
            hoverinfo='text',
            marker=dict(
                color=node_color,
                colorscale='Viridis',
                size=node_size,
                showscale=True,
                colorbar=dict(
                    title='分数',
                    thickness=15
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # 简化的布局
        layout = dict(
            title='ToT搜索树',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        fig.write_html(output_file)
        print(f"✅ 简化版可视化已保存到: {output_file}")
        
        return fig
        
    except Exception as e:
        print(f"❌ 简化可视化失败: {e}")
        return None


def visualize_search_tree_text(tree_data: Dict[str, Any]):
    """
    文本格式的搜索树可视化（无依赖）
    
    Args:
        tree_data: 搜索树数据
    """
    print("\n" + "="*70)
    print("🌳 搜索树文本可视化")
    print("="*70)
    
    nodes = tree_data["nodes"]
    edges = tree_data["edges"]
    
    # 构建父子关系
    children = {}
    for edge in edges:
        parent = edge["from"]
        child = edge["to"]
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    
    # 按分数排序子节点
    for parent in children:
        children[parent].sort(
            key=lambda c: nodes.get(c, {}).get("score", 0), 
            reverse=True
        )
    
    def print_node(node_id: str, depth: int = 0, prefix: str = "", is_last: bool = False):
        """递归打印节点"""
        if depth > 3:  # 限制显示深度
            if node_id in children:
                print(f"{prefix}└── ... (还有{len(children[node_id])}个子节点)")
            return
        
        node_info = nodes.get(node_id, {})
        state = node_info.get("state", [])
        score = node_info.get("score", 0.0)
        visits = node_info.get("visits", 0)
        
        # 节点表示
        if state:
            state_str = " → ".join(state[:3])  # 只显示前3个组件
            if len(state) > 3:
                state_str += f" ... (+{len(state)-3})"
        else:
            state_str = "[根]"
        
        # 节点标记
        if score >= 0.8:
            marker = "★"
        elif score >= 0.6:
            marker = "●"
        else:
            marker = "○"
        
        # 打印节点
        if depth == 0:
            print(f"{marker} {state_str}")
            print(f"   📊 分数: {score:.3f} | 👁️ 访问: {visits}")
        else:
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{marker} {state_str}")
            print(f"{prefix}    📊 {score:.3f} | 👁️ {visits}")
        
        # 打印子节点
        if node_id in children:
            new_prefix = prefix + ("    " if is_last else "│   ")
            child_count = len(children[node_id])
            
            for i, child_id in enumerate(children[node_id][:3]):  # 最多显示3个子节点
                child_is_last = (i == min(2, child_count-1))
                print_node(child_id, depth + 1, new_prefix, child_is_last)
            
            # 如果有更多子节点
            if child_count > 3:
                print(f"{new_prefix}└── ... 还有 {child_count-3} 个子节点")
    
    # 从根节点开始打印
    print_node("root")
    
    # 显示统计信息
    print("\n📊 搜索树统计:")
    print(f"   📍 总节点数: {len(nodes)}")
    print(f"   🔗 总边数: {len(edges)}")
    
    # 计算深度
    depths = [info.get("depth", 0) for info in nodes.values()]
    if depths:
        print(f"   📏 最大深度: {max(depths)}")
        print(f"   📏 平均深度: {sum(depths)/len(depths):.1f}")
    
    # 显示最佳节点
    best_nodes = sorted(
        nodes.items(), 
        key=lambda x: x[1].get("score", 0), 
        reverse=True
    )[:5]
    
    print("\n🏆 最佳节点:")
    for i, (node_id, info) in enumerate(best_nodes):
        state = info.get("state", [])
        state_str = " → ".join(state) if state else "[根]"
        score = info.get("score", 0.0)
        visits = info.get("visits", 0)
        
        # 缩短长状态
        if len(state_str) > 40:
            state_str = state_str[:37] + "..."
        
        print(f"   {i+1}. {state_str}")
        print(f"      📊 {score:.3f} | 👁️ {visits}")
    
    print("="*70)


def save_search_tree_json(tree_data: Dict[str, Any], output_file: str = "search_tree.json"):
    """
    保存搜索树为JSON文件
    
    Args:
        tree_data: 搜索树数据
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 搜索树数据已保存到: {output_file}")
    except Exception as e:
        print(f"❌ 保存JSON失败: {e}")


if __name__ == "__main__":
    # 测试代码
    sample_tree = {
        "root": "root",
        "nodes": {
            "root": {
                "state": [],
                "score": 0.5,
                "depth": 0,
                "visits": 10
            },
            "node1": {
                "state": ["conciseness"],
                "score": 0.7,
                "depth": 1,
                "visits": 5
            },
            "node2": {
                "state": ["three_points"],
                "score": 0.8,
                "depth": 1,
                "visits": 8
            }
        },
        "edges": [
            {"from": "root", "to": "node1"},
            {"from": "root", "to": "node2"}
        ]
    }
    
    # 测试文本可视化
    visualize_search_tree_text(sample_tree)
    
    # 测试图形可视化
    try:
        fig = visualize_search_tree(sample_tree, "test_tree.html")
        if fig:
            print("✅ 图形可视化测试成功")
    except:
        print("⚠️  图形可视化测试失败，使用文本可视化")