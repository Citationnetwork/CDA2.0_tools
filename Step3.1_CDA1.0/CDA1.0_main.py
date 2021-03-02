
import json
import random
import plotly.graph_objs as go
import plotly.io as pio

if __name__ == '__main__':

    # 生成节点颜色，结果储存在 color_nodes
    # 生成簇团内连线颜色，结果储存在 color_lines
    # 将团间颜色设置为背景色
    def color_setting(num_node, num_edge, comm, links, links_cluster):
        color_node = [0] * num_node
        color_cluster = []
        for i in range(len(comm)):
            color = 'rgb({},{},{})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            color_cluster.append(color)
            for j in range(len(comm[i])):
                color_node[comm[i][j]] = color
        color_line = [0] * 3 * num_edge
        for i in range(num_edge):
            for j in range(len(comm)):
                if links[i] in links_cluster[j]:
                    color_line[3*i] = color_cluster[j]
                    color_line[3*i+1] = color_cluster[j]
        for i in range(num_edges):
            if color_line[3*i] == 0:
                color_line[3*i] = 'rgba(0,0,0,0)'
                color_line[3*i+1] = 'rgba(0,0,0,0)'
        return color_node, color_line

    # 坐标处理
    def coordinates_process(coordinate, links):
        xn = [coordinate[k][0] for k in range(num_nodes)]  # x-coordinates of nodes
        yn = [coordinate[k][1] for k in range(num_nodes)]  # y-coordinates of nodes
        zn = [0] * num_nodes
        xe, ye = [], []
        for edge in edges:
            xe += [coordinate[edge[0]][0], coordinate[edge[1]][0], None]  # x-coordinates of edge ends
            ye += [coordinate[edge[0]][1], coordinate[edge[1]][1], None]  # y-coordinates of edge ends
        ze = [0, 0, None] * len(links)
        return xn, yn, zn, xe, ye, ze

    data_path_json = './data/patent_top10_input.json'   # 输入文件路径
    output_path = './result/patent_top10_input.png'      # 输出文件路径
    node_size = 1                              # 节点尺寸大小调节
    line_width = 0.8                           # 线条直径大小

    # 读取数据
    f = open(data_path_json, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # 获取数据规模
    num_nodes = len(data['nodes'])
    num_edges = len(data['links'])
    print('节点数：', num_nodes, '连接数：', num_edges)

    # 读取节点姓名，连接及权重信息
    labels = [data['nodes'][k]['name'] for k in range(num_nodes)]
    edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(num_edges)]
    nodes_coordinates = [data['coordinates'][k]['coordinate'] for k in range(num_nodes)]
    clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]
    edges_cluster = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]
    for i in range(len(edges_cluster)):
        for j in range(len(edges_cluster[i])):
            edges_cluster[i][j] = tuple(edges_cluster[i][j])

    #设置节点、线条颜色
    color_nodes, color_lines = color_setting(num_node=num_nodes, num_edge=num_edges, comm=clusters,
                                             links=edges, links_cluster=edges_cluster)

    #坐标处理
    Xn, Yn, Zn, Xe, Ye, Ze = coordinates_process(coordinate=nodes_coordinates, links=edges)

    #可视化呈现
    trace_line = go.Scatter3d(x=Xe,
                              y=Ye,
                              z=Ze,
                              mode='lines',
                              line=dict(color=color_lines, width=line_width),  # 线的颜色与尺寸
                              hoverinfo='none'
                              )

    trace_node = go.Scatter3d(x=Xn,
                              y=Yn,
                              z=Zn,
                              mode='markers',
                              name='actors',
                              marker=dict(symbol='circle',
                                          size=node_size,  # size of nodes
                                          color=color_nodes,  # color of nodes
                                          colorscale='Viridis'
                                          ),
                              text=labels,
                              hoverinfo='text'
                              )

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(title="CDA2.0",
                       width=1500,
                       height=1500,
                       showlegend=True,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis),
                                  ),
                       margin=dict(t=100),
                       hovermode='closest',
                       annotations=[dict(showarrow=False,
                                         text='节点数：' + str(num_nodes) + '连线数：' + str(num_edges),
                                         xref='paper',
                                         yref='paper',
                                         x=0,
                                         y=0.1,
                                         xanchor='left',
                                         yanchor='bottom',
                                         font=dict(size=14)
                                         )
                                    ]
                       )

    data = [trace_line, trace_node]
    fig = go.Figure(data=data, layout=layout)

    # 生成 html 格式文件
    #fig.write_html(output_path, auto_open=False)
    fig.write_image(output_path)
