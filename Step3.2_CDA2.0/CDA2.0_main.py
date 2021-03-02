
import json
import random
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

if __name__ == '__main__':

    # 读取json文件
    def read_json(data_path):
        f = open(data_path, 'r', encoding='utf-8')
        data_dict = json.loads(f.read())
        f.close()
        return data_dict

    # 读取数据详细信息
    def data_collation(data):
        num_nodes = len(data['nodes'])
        num_edges = len(data['links'])
        label = [data['nodes'][k]['name'] for k in range(len(data['nodes']))]
        edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(len(data['links']))]
        nodes_coordinates = [data['coordinates'][k]['coordinate'] for k in range(len(data['nodes']))]
        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]
        edges_cluster = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]
        for i in range(len(edges_cluster)):
            for j in range(len(edges_cluster[i])):
                edges_cluster[i][j] = tuple(edges_cluster[i][j])
        return num_nodes, num_edges, label, edges, nodes_coordinates, clusters, edges_cluster

    # 生成节点颜色，结果储存在 color_nodes
    # 生成簇团内连线颜色，结果储存在 color_lines
    # 将团间颜色设置为透明色
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
        for i in range(num_edge):
            if color_line[3*i] == 0:
                color_line[3*i] = 'rgba(0,0,0,0)'  # 透明色
                color_line[3*i+1] = 'rgba(0,0,0,0)'
        return color_node, color_line

    # 坐标调整，将网络重心调整到(0,0)
    def coordinates_adjustment(coordinate):
        n = len(coordinate)
        x_center = sum(i[0] for i in coordinate)/n
        y_center = sum(i[1] for i in coordinate)/n
        for i in coordinate:
            i[0] -= x_center
            i[1] -= y_center
        return coordinate

    # 坐标跨度计算，计算节点坐标在 x,y 两个维度上的跨度
    def span_calculation(coordinate):
        x_max = max(i[0] for i in coordinate)
        y_max = max(i[1] for i in coordinate)
        x_min = min(i[0] for i in coordinate)
        y_min = min(i[1] for i in coordinate)
        x_span = x_max-x_min
        y_span = y_max-y_min
        return x_span, y_span

    # 双层坐标跨度匹配及层间高度计算，将坐标跨度小的一层向坐标跨度大的一层放大
    # 并将层高设定为与坐标跨度相等
    def coordinate_match(coordinate_1, coordinate_2):
        x_span_1, y_span_1 = span_calculation(coordinate_1)
        x_span_2, y_span_2 = span_calculation(coordinate_2)
        k = ((x_span_1*y_span_1)/(x_span_2*y_span_2))**0.5
        if k > 1:
            for coordinate in coordinate_2:
                coordinate[0] *= k
                coordinate[1] *= k
            high = (x_span_1*y_span_1)**0.5
        else:
            for coordinate in coordinate_1:
                coordinate[0] /= k
                coordinate[1] /= k
            high = (x_span_2 * y_span_2) ** 0.5
        return coordinate_1, coordinate_2, high

    # 双层同名节点匹配
    def same_nodes_match(label_1, label_2):
        edges = []
        name_same = []
        for i in range(len(label_1)):
            for j in range(len(label_2)):
                if label_1[i] == label_2[j]:
                    edges.append([i, j])
                    name_same.append(label_1[i])
        return edges, name_same

    # 将簇团的划分的id列表转换为机构名称的集合
    def get_namelist(name, comm):
        clusters_name = [set() for i in range(len(comm))]
        for i in range(len(comm)):
            for nodeid in comm[i]:
                clusters_name[i].add(name[nodeid])
        return clusters_name

    def find_max_degree(g_edge):
        g_nx = nx.Graph()
        g_nx.add_edges_from(g_edge)
        degree = dict(g_nx.degree())
        degree_max = max(degree, key=lambda x: degree[x])
        return degree_max

    # 双层社区相似度匹配
    def layout_link(label_1, label_2, comm_1, comm_2, edge_1, edge_2):
        name_1 = get_namelist(name=label_1, comm=comm_1)
        name_2 = get_namelist(name=label_2, comm=comm_2)
        edges = []
        id_link = []
        num_1, num_2 = len(comm_1), len(comm_2)
        for i in range(num_1):
            for j in range(num_2):
                similar = float(len(name_1[i] & name_2[j])) / len(name_1[i] | name_2[j])
                if (similar >= 0.01) and len(name_1[i] | name_2[j]) >= 100:
                    id_link.append((i, j))
                    edges.append((find_max_degree(edge_1[i]), find_max_degree(edge_2[j])))

        return edges, id_link

    # 坐标处理，生成单维度坐标列表
    def coordinates_process(coordinate_1, coordinate_2, links_1, links_2, links_3, height):
        n_1, n_2 = len(coordinate_1), len(coordinate_2)
        l_1, l_2, l_3 = len(links_1), len(links_2), len(links_2)
        xn = [coordinate_1[k][0] for k in range(n_1)] + [coordinate_2[k][0] for k in range(n_2)]
        yn = [coordinate_1[k][1] for k in range(n_1)] + [coordinate_2[k][1] for k in range(n_2)]
        zn = [0]*n_1 + [height]*n_2
        xe, ye = [], []
        for link in links_1:
            xe += [coordinate_1[link[0]][0], coordinate_1[link[1]][0], None]  # x-coordinates of edge ends
            ye += [coordinate_1[link[0]][1], coordinate_1[link[1]][1], None]  # y-coordinates of edge ends
        for link in links_2:
            xe += [coordinate_2[link[0]][0], coordinate_2[link[1]][0], None]  # x-coordinates of edge ends
            ye += [coordinate_2[link[0]][1], coordinate_2[link[1]][1], None]  # y-coordinates of edge ends
        for link in links_3:
            xe += [coordinate_1[link[0]][0], coordinate_2[link[1]][0], None]  # x-coordinates of edge ends
            ye += [coordinate_1[link[0]][1], coordinate_2[link[1]][1], None]  # y-coordinates of edge ends
        ze = [0, 0, None]*l_1 + [height, height, None]*l_2 + [0, height, None]*l_3
        return xn, yn, zn, xe, ye, ze

    # 程序的七个输入
    data_path_1 = './data/invoic_2.json'            # 下层文件读取路径
    data_path_2 = './data/patent_2.json'    # 上层文件读取路径
    output_path = './result/patent_invoic_louvain_2.png'   # 结果输出路径
    linkways_switch = 1   #可供选择参数是0或1，双层连接方式选择，0代表同名节点连接，1代表簇团相似度连接
    node_size = 0.5       # 节点尺寸大小
    line_width_1 = 0.5    # 层内线条直径大小
    line_width_2 = 0.8     # 层间线条直径大小

    data_1 = read_json(data_path_1)
    data_2 = read_json(data_path_2)

    num_nodes_1, num_edges_1, labels_1, edges_1, nodes_coordinates_1, clusters_1, edges_cluster_1 = data_collation(data_1)
    num_nodes_2, num_edges_2, labels_2, edges_2, nodes_coordinates_2, clusters_2, edges_cluster_2 = data_collation(data_2)
    print('层 1 节点数：', num_nodes_1, '层 1 连接数：', num_edges_1)
    print('层 2 节点数：', num_nodes_2, '层 2 连接数：', num_edges_2)
    labels = labels_1 + labels_2
    print("1--数据读取完毕！")

    color_nodes_1, color_lines_1 = color_setting(num_node=num_nodes_1, num_edge=num_edges_1,
                                                 comm=clusters_1, links=edges_1, links_cluster=edges_cluster_1)
    print("2--第一层颜色设置完毕！")

    color_nodes_2, color_lines_2 = color_setting(num_node=num_nodes_2, num_edge=num_edges_2,
                                                 comm=clusters_2, links=edges_2, links_cluster=edges_cluster_2)
    print("3--第二层颜色设置完毕！")

    nodes_coordinates_1 = coordinates_adjustment(coordinate=nodes_coordinates_1)
    nodes_coordinates_2 = coordinates_adjustment(coordinate=nodes_coordinates_2)
    print("4--双层的坐标调整完毕！")

    nodes_coordinates_1, nodes_coordinates_2, h = coordinate_match(coordinate_1=nodes_coordinates_1,
                                                                   coordinate_2=nodes_coordinates_2)
    print("5--坐标匹配完成！")

    #连接方式选择
    if linkways_switch == 0:
        edges_3, labels_same = same_nodes_match(label_1=labels_1, label_2=labels_2)
        num_edges_3 = len(edges_3)
        print(edges_3)
    elif linkways_switch == 1:
        edges_3, cluster_link = layout_link(label_1=labels_1, label_2=labels_2,
                                            comm_1=clusters_1, comm_2=clusters_2,
                                            edge_1=edges_cluster_1, edge_2=edges_cluster_2)
        num_edges_3 = len(edges_3)
        print(edges_3)




    color_nodes = color_nodes_1 + color_nodes_2
    color_lines = color_lines_1 + color_lines_2 + ['rgb(100,100,100)', 'rgb(100,100,100)', 0]*num_edges_3



    Xn, Yn, Zn, Xe, Ye, Ze = coordinates_process(coordinate_1=nodes_coordinates_1,
                                                 coordinate_2=nodes_coordinates_2,
                                                 links_1=edges_1,
                                                 links_2=edges_2,
                                                 links_3=edges_3,
                                                 height=h)

    trace_line_1 = go.Scatter3d(x=Xe[0:-3*num_edges_3], y=Ye[0:-3*num_edges_3], z=Ze[0:-3*num_edges_3],
                                mode='lines',
                                line=dict(color=color_lines[0:-3*num_edges_3], width=line_width_1),  # 线的颜色与尺寸
                                hoverinfo='none'
                                )

    trace_line_2 = go.Scatter3d(x=Xe[-3*num_edges_3:], y=Ye[-3*num_edges_3:], z=Ze[-3*num_edges_3:],
                                mode='lines',
                                line=dict(color=color_lines[-3*num_edges_3:], width=line_width_2),  # 线的颜色与尺寸
                                hoverinfo='none'
                                )

    trace_node = go.Scatter3d(x=Xn, y=Yn, z=Zn,
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
                                         text='上层节点数：' + str(num_nodes_1) + '<br>' +
                                              '上层连线数：' + str(num_edges_1) + '<br>' +
                                              '下层节点数：' + str(num_nodes_1) + '<br>' +
                                              '下层连线数：' + str(num_edges_1) + '<br>' +
                                              '层间连线数：' + str(len(edges_3)),
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

    data_plot = [trace_line_1, trace_line_2, trace_node]
    fig = go.Figure(data=data_plot, layout=layout)

    # 生成 html 格式文件
    #fig.write_html(output_path, auto_open=False)

    # 报错时 请在cmd中输入 conda install -c plotly plotly-orca
    #
    #fig.write_image(output_path)
    pio.write_image(fig,output_path)
