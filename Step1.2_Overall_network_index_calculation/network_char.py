import os
import json
import networkx as nx
import xlsxwriter as xw


# 取字典value的平均值，dict_1表示传进的参数是字典
def average_value(dict_1):
    return sum(dict_1.values()) / len(dict_1)


# 对字典按值进行排序，取前k，并返回字典
def get_k(dict_1, k):
    return dict(sorted(dict_1.items(), key=lambda x: x[1], reverse=True)[0:k])


# 求出两个字典的值的积
def mul(dict_1, dict_2):
    return dict(zip(list(dict_1.keys()), list(map(lambda x, y: x * y, list(dict_1.values()), list(dict_2.values())))))


def fun1(R):
    R_max = max(R)
    sum_R = 0.0
    for R_i in R:
        sum_R = sum_R + R_i
    n = len(R)

    return R_max * n - sum_R, n


def C_CP_RD(RD):
    sum_sub, n = fun1(RD.values())
    return sum_sub / (n - 2)


def C_CP_RB(RB):
    sum_sub, n = fun1(RB.values())
    return sum_sub / (n - 1)


def C_CP_RP(RP):
    sum_sub, n = fun1(RP.values())
    return sum_sub / (n - 1) / (n - 2) * (2 * n - 3)


def statistic_analysis(G, output):
    write_book = xw.Workbook(output)
    # 总表
    write_book_sheet1 = write_book.add_worksheet('总表')
    col = [['名称', '值'],
           ['节点数量', G.number_of_nodes()],
           ['边数量', G.number_of_edges()],
           ['网络密度', nx.density(G)],
           ['网络传递性', nx.transitivity(G)],
           ['平均聚类系数', average_value(nx.clustering(G))],
           ['平均点度中心度（相对）', average_value(nx.degree_centrality(G))],
           ['平均接近中心度', average_value(nx.closeness_centrality(G))],
           ['平均中介中心度', average_value(nx.betweenness_centrality(G))],
           # ['平均特征向量中心度', average_value(nx.eigenvector_centrality(G))],
           # ['average_efficiency', average_value(nx.effective_size(G))],
           ['average_constraint', average_value(nx.constraint(G))],
           ['average_embeddedness', average_value(mul(nx.betweenness_centrality(G), nx.clustering(G)))],
           ['点度中心势（相对）', C_CP_RD(nx.degree_centrality(G))],
           ['中介中心势', C_CP_RB(nx.betweenness_centrality(G))],
           ['接近中心势', C_CP_RP(nx.closeness_centrality(G))]
           ]  # 算法见论文公式表
    for i in range(0, len(col)):
        write_book_sheet1.write(i, 0, col[i][0])
        write_book_sheet1.write(i, 1, col[i][1])
    print('sum_gheet done!')

    # 输出排名数目
    K = int(G.number_of_nodes() / 1)
    # 写一个二维数组装下表名和字典
    sheet_box = [['聚类系数', nx.clustering(G)],
                 ['点度中心度（相对）', nx.degree_centrality(G)],
                 ['接近中心度', nx.closeness_centrality(G)],
                 ['中介中心度', nx.betweenness_centrality(G)],
                 # ['特征向量中心度', nx.eigenvector_centrality(G)],
                 # ['efficiency', nx.effective_size(G)],
                 # ['constraint', nx.constraint(G)],
                 # ['embeddedness', mul(nx.betweenness_centrality(G), nx.clustering(G))]
                 ]  # 算法见论文公式表
    for i in range(0, len(sheet_box)):
        write_book_sheet2 = write_book.add_worksheet(sheet_box[i][0])
        write_book_sheet2.write(0, 0, '排名')
        write_book_sheet2.write(0, 1, '名称')
        write_book_sheet2.write(0, 2, '值')
        output_dict = get_k(sheet_box[i][1], K)
        for j in range(0, K):
            write_book_sheet2.write(j + 1, 0, j + 1)
            write_book_sheet2.write(j + 1, 1, list(output_dict.keys())[j])
            write_book_sheet2.write(j + 1, 2, list(output_dict.values())[j])
        print('sheet ' + str(i + 1) + ' done!')

    print(output + ' done!')
    write_book.close()


if __name__ == '__main__':
    json_name = '引用网络_示例.json'   #输入数据文件名

    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_整体网络输入和输出数据\\输入数据_excel', json_name)
    graph_inf = json.load(open(json_path, 'r', encoding='UTF-8'))
    nodes = []
    for each_node in graph_inf['nodes']:
        nodes.append(each_node['name'])
    links = []
    for each_link in graph_inf['links']:
        links.append([nodes[int(each_link['source'])], nodes[int(each_link['target'])]])

    G = nx.Graph()
    G.add_edges_from(links)
    excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_整体网络输入和输出数据\\输出数据_excel',
                              os.path.splitext(json_name)[0] + '.xlsx')
    statistic_analysis(G, excel_path)
