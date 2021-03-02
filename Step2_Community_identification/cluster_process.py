import json
import igraph as ig

'''
增加了簇团识别之后json文件中cluster中的连接权重信息weights
'''


if __name__ == '__main__':

    # 社区发现函数
    # 给定algorithm不同值选择不同社区发现算法
    # 1-newman  2-louvain  3-leading_eigenvector  4-label_propagation
    # 5-walktrap  6-GN  7-spinglass  8-infomap
    def comm_detection(g, algorithm, weights_input):
        if algorithm == 1:
            # (1)社区发现，newman快速算法，VertexDendrogram
            comm = list(g.community_fastgreedy(weights=weights_input).as_clustering())
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 2:
            # (2)社区发现，louvain算法，VertexClustering
            comm = list(g.community_multilevel(weights=weights_input))
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 3:
            # (3)社区发现，leading_eigenvector算法，VertexClustering
            comm = list(g.community_leading_eigenvector(weights=weights_input))
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 4:
            # (4)社区发现，label_propagation算法，VertexClustering
            comm = list(g.community_label_propagation(weights=weights_input, initial=None, fixed=None))
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 5:
            # (5)社区发现，walktrap算法，VertexDendrogram
            comm = list(g.community_walktrap(weights=weights_input, steps=4).as_clustering())
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 6:
            # (6)社区发现，GN算法，VertexDendrogram(时间久，不推荐大网络)
            se = g.community_edge_betweenness(clusters=None, directed=False, weights=weights_input)
            n = se._graph.vcount()
            max_q, optimal_count = 0, 1
            for step in range(min(n - 1, len(se._merges))):
                membs = ig.community_to_membership(se._merges, n, step)
                q = se._graph.modularity(membs, **se._modularity_params)
                if q > max_q:
                    optimal_count = n - step
                    max_q = q
            se._optimal_count = optimal_count
            comm = list(se.as_clustering())
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 7:
            # (7)社区发现，spinglass算法，VertexClustering(时间久，不推荐大网络)
            clusters = g.clusters()
            giant = clusters.giant()
            comm = list(giant.community_spinglass())
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)

        elif algorithm == 8:
            # (8)社区发现，infomap算法，VertexClustering
            # 本算法需要给定簇团划分数量(trials=10)，初始值为10
            comm = list(g.community_infomap(edge_weights=weights_input, vertex_weights=None, trials=10))
            comm.sort(key=lambda i: len(i), reverse=True)
            num = len(comm)
        else:
            print('算法指定错误')

        return comm, num

    # 二次聚类
    def multiple_clustering(comm, edge_weight, max_n):
        j = 0
        while len(comm[j]) > max_n:
            max_cluster_last = len(comm[j])
            node_dict = {}
            node_dict_reverse = {}
            edges_0, weight_0 = [], []
            for i in range(len(comm[0])):
                node_dict[comm[0][i]] = i
                node_dict_reverse[i] = comm[0][i]
            for ew in edge_weight:
                if (ew[0] in comm[0]) and (ew[1] in comm[0]):
                    edges_0.append((node_dict[ew[0]], node_dict[ew[1]]))
                    weight_0.append(ew[2])
            g_0 = ig.Graph(edges_0, directed=False)
            comm_0, num_comm_0 = comm_detection(g=g_0, algorithm=k_algorithm, weights_input=weight_0)
            comm_0_reverse = [[] for i in range(num_comm_0)]
            for i in range(num_comm_0):
                for node in comm_0[i]:
                    comm_0_reverse[i].append(node_dict_reverse[node])
            del comm[0]
            comm += comm_0_reverse
            comm.sort(key=lambda i: len(i), reverse=True)
            if max_cluster_last == len(comm[j]):
                j += 1
        return comm

    # 簇团内连接及簇团间连接分类
    def edges_sort(comm, links):
        edge_cluster = [[] for i in range(len(comm))]
        edges_in, edges_between = [], []
        for link in links:
            for i in range(len(comm)):
                if (link[0] in comm[i]) and (link[1] in comm[i]):
                    edges_in.append(link)
                    edge_cluster[i].append(link)
                else:
                    edges_between.append(link)
        return edges_in, edges_between, edge_cluster

    # 将全部结果写入JSON文件
    def write_json(main_data, coordinates, clusters, edges_cluster, cluster_weight, output_file_path):
        dict_nodes, dict_clusters, dict_coordinates = [], [], []
        for i in range(len(main_data['nodes'])):
            dict_nodes.append({'node_id': i, 'name': main_data['nodes'][i]['name']})
        for i in range(len(coordinates)):
            dict_coordinates.append({'node': i, 'coordinate': coordinates[i]})
        for i in range(len(clusters)):
            dict_clusters.append({'cluster_id': i, 'cluster': clusters[i], 'edges': edges_cluster[i],
                                  'weights': cluster_weight[i]})

        inf_dict = {'nodes': dict_nodes,
                    'links': main_data['links'],
                    'coordinates': dict_coordinates,
                    'clusters': dict_clusters}

        output_file = open(output_file_path, 'w', encoding='UTF-8')
        json.dump(inf_dict, output_file, ensure_ascii=False)

        output_file = open(output_file_path, 'r', encoding='UTF-8')
        json_str = str(output_file.read()).replace('}, ', '},\n').replace('}], "', '}],\n"').replace(' ', '')

        output_file = open(output_file_path, 'w', encoding='UTF-8')
        output_file.write(json_str)
        output_file.close()

    # 函数所需要的三个输入量和一个人为设定的参数
    data_path_json = './input_data/invoic.json'     # json文件读取路径
    output_path = './result/invoic_222_newman.json'    # json文件保存路径
    k_algorithm = 2      # 给定k_algorithm不同值选择不同社区发现算法
    max_nodes = 100000    # 社团最大点数
    # 1-newman    2-louvain  3-leading_eigenvector  4-label_propagation
    # 5-walktrap  6-GN       7-spinglass            8-infomap

    # 读取数据
    f = open(data_path_json, 'r', encoding='utf-8')
    data = json.loads(f.read())
    f.close()

    # 获取数据规模
    num_nodes = len(data['nodes'])
    num_edges = len(data['links'])
    print('节点数：', num_nodes, '连接数：', num_edges)

    # 读取节点姓名，连接及权重信息
    edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(num_edges)]
    weights = [(data['links'][k]['value']) for k in range(num_edges)]
    edges_weights = [(data['links'][k]['source'],
                      data['links'][k]['target'],
                      data['links'][k]['value']) for k in range(num_edges)]

    # 建立一个Graph，并添加节点，连接信息
    G_ig = ig.Graph()
    G_ig.add_vertices(num_nodes)
    G_ig = ig.Graph(edges, directed=False)

    #开始社区划分
    result, num_comm = comm_detection(g=G_ig, algorithm=k_algorithm, weights_input=weights)
    print(len(result[0]), len(result[1]), len(result[2]))

    #开始多次聚类
    result = multiple_clustering(comm=result, edge_weight=edges_weights, max_n=max_nodes)
    print(len(result[0]), len(result[1]), len(result[2]))

    #开始簇团信息的分类，簇团间和簇团内以及所有连接信息
    edges_inside, edges_outside, edges_comm = edges_sort(comm=result, links=edges)

    # 建立一个Graph，并添加节点及簇团内的连接信息
    #开始力引导布局，只针对簇团内的连接信息，这样可以将簇团更好地区分开，可视化效果好
    G_part = ig.Graph()
    G_part.add_vertices(num_nodes)
    G_part = ig.Graph(edges_inside, directed=False)
    nodes_coordinates = G_part.layout('fr')

    #增加"cluster"信息中的  边的权重信息
    clusters_weights = []
    for i in range(len(edges_comm)):
        clusters_weights.append([])
        for j in range(len(edges_weights)):
            if (edges_weights[j][0], edges_weights[j][1]) in edges_comm[i]:
                clusters_weights[i].append([edges_weights[j][0], edges_weights[j][1], edges_weights[j][2]])

    write_json(main_data=data, coordinates=nodes_coordinates, clusters=result,
               edges_cluster=edges_comm, cluster_weight = clusters_weights, output_file_path=output_path)
