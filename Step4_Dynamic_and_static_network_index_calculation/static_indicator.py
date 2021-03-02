#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
"""
# @version : V1.0
# @Author  : mxz
# @contact : 925813094@qq.com
# @Time    : 2020.3.2
# @File    : static_indicator.py
# @Software: PyCharm
"""
# 计算子群静态指标
# JSON 文件需按年份有序命名，文件名的前四位必须为年份，确保程序能正常处理文件
# 文件路径为该程序路径下的data文件夹

import os
import math
import json
import numpy as np
import pandas as pd
import networkx as nx


def fun1(r_values):
    r_max = max(r_values)
    sum_r = 0.0
    for r_i in r_values:
        sum_r =+ r_i
    n_r = len(r_values)

    return r_max * n_r - sum_r


def write_excel(writer, array, ind, col, sheet):
    df = pd.DataFrame(array, index=ind, columns=col)
    df.to_excel(writer, sheet_name=sheet)


# degree centrality、betweenness centrality、closeness centrality、约束指数（constraint）
# # 节点所在子群的指标：每个子群density、三种中心势（centralization）
# 跨子群连接数量，子群的模块度
def static_indicators(path, file_name, num_firm, writer):
    density_array = np.zeros((len(file_name), num_firm))
    constraint_array = np.zeros((len(file_name), num_firm))
    degree_centrality_array = np.zeros((len(file_name), num_firm))
    betweenness_centrality_array = np.zeros((len(file_name), num_firm))
    closeness_centrality_array = np.zeros((len(file_name), num_firm))
    degree_centralization_array = np.zeros((len(file_name), num_firm))
    betweenness_centralization_array = np.zeros((len(file_name), num_firm))
    closeness_centralization_array = np.zeros((len(file_name), num_firm))
    bridging_ties_array = np.zeros((len(file_name), num_firm))
    modularity_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        degree_part_dict = dict()
        density_dict = dict()
        constraint_dict = dict()
        degree_centrality_dict = dict()
        betweenness_centrality_dict = dict()
        closeness_centrality_dict = dict()
        degree_centralization_dict = dict()
        betweenness_centralization_dict = dict()
        closeness_centralization_dict = dict()
        modularity_dict = dict()
        years.append(file_name[name][0:-5])
        # 文件路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        num_edges = len(data['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(num_edges)]
        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]
        clusters_edges = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]

        g_full = nx.Graph()
        g_full.add_nodes_from(list(range(num_firm)))
        g_full.add_edges_from(edges)

        degree_dict = dict(nx.degree(g_full))

        for i in range(len(clusters)):
            n_cluster = len(clusters[i])
            g = nx.Graph()
            g.add_nodes_from(clusters[i])
            g.add_edges_from(clusters_edges[i])

            degree_cluster = dict(nx.degree(g))
            sum_degree = sum(degree_cluster.values())

            degree_part_dict = {**degree_part_dict, **degree_cluster}

            density_cluster = nx.density(g)
            for firm in clusters[i]:
                density_dict[firm] = density_cluster

            constraint_cluster = nx.constraint(g)
            constraint_dict = {**constraint_dict, **constraint_cluster}

            degree_centrality_cluster = nx.degree_centrality(g)
            degree_centrality_dict = {**degree_centrality_dict, **degree_centrality_cluster}

            betweenness_centrality_cluster = nx.betweenness_centrality(g)
            betweenness_centrality_dict = {**betweenness_centrality_dict, **betweenness_centrality_cluster}

            closeness_centrality_cluster = nx.closeness_centrality(g)
            closeness_centrality_dict = {**closeness_centrality_dict, **closeness_centrality_cluster}
            if n_cluster == 1 or n_cluster == 2:
                for firm in clusters[i]:
                    degree_centralization_dict[firm] = 0
                    betweenness_centralization_dict[firm] = 0
                    closeness_centralization_dict[firm] = 0
            else:
                dc = fun1(degree_centrality_cluster.values())
                bc = fun1(betweenness_centrality_cluster.values())
                cc = fun1(closeness_centrality_cluster.values())
                degree_centralization_cluster = dc / (n_cluster - 2)
                betweenness_centralization_cluster = bc / (n_cluster - 1)
                closeness_centrality_cluster = cc * (2*n_cluster - 3) / ((n_cluster-2)*(n_cluster-1))
                for firm in clusters[i]:
                    degree_centralization_dict[firm] = degree_centralization_cluster
                    betweenness_centralization_dict[firm] = betweenness_centralization_cluster
                    closeness_centralization_dict[firm] = closeness_centrality_cluster

            # 子群的模块度
            modularity = 0
            cluster = clusters[i]
            if n_cluster == 1:
                modularity = None
                modularity_dict[cluster[0]] = None
            else:
                for firm_i in range(len(cluster)):
                    for firm_j in range(firm_i+1, len(cluster)):
                        i_j = degree_cluster[cluster[firm_i]]*degree_cluster[cluster[firm_j]]/sum_degree
                        if [cluster[firm_i], cluster[firm_j]] in clusters_edges[i]:
                            modularity += (1 - i_j)/sum_degree
                        else:
                            modularity += (0 - i_j)/sum_degree

            for firm in clusters[i]:
                modularity_dict[firm] = modularity

        for i in range(num_firm):
            density_array[name, i] = density_dict[i]
            constraint_array[name, i] = constraint_dict[i]
            degree_centrality_array[name, i] = degree_centrality_dict[i]
            betweenness_centrality_array[name, i] = betweenness_centrality_dict[i]
            closeness_centrality_array[name, i] = closeness_centrality_dict[i]
            degree_centralization_array[name, i] = degree_centralization_dict[i]
            betweenness_centralization_array[name, i] = betweenness_centralization_dict[i]
            closeness_centralization_array[name, i] = closeness_centralization_dict[i]
            bridging_ties_array[name, i] = degree_dict[i] - degree_part_dict[i]
            modularity_array[name, i] = modularity_dict[i]

    write_excel(writer, density_array, years, list(range(num_firm)), '网络密度')
    write_excel(writer, constraint_array, years, list(range(num_firm)), 'Constraint')
    write_excel(writer, degree_centrality_array, years, list(range(num_firm)), '点度中心度')
    write_excel(writer, betweenness_centrality_array, years, list(range(num_firm)), '中介中心度')
    write_excel(writer, closeness_centrality_array, years, list(range(num_firm)), '接近中心度')
    write_excel(writer, degree_centralization_array, years, list(range(num_firm)), '点度中心势')
    write_excel(writer, betweenness_centralization_array, years, list(range(num_firm)), '中介中心势')
    write_excel(writer, closeness_centralization_array, years, list(range(num_firm)), '接近中心势')
    write_excel(writer, bridging_ties_array, years, list(range(num_firm)), 'bridging_ties')
    write_excel(writer, modularity_array, years, list(range(num_firm)), '子群模块度')
    print('十大静态指标计算完成')


# 计算参与系数
def participation_coefficient(path, file_name, num_firm, writer):
    # 创建参与系数矩阵（年份×公司数）
    p_c_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件读取路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        # 获取数据规模
        num_nodes = len(data['nodes'])
        num_edges = len(data['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(num_edges)]
        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]

        # 创建整个网络完整 Graph
        g = nx.Graph()
        g.add_nodes_from(list(range(num_nodes)))
        g.add_edges_from(edges)
        degree_dict = nx.degree(g)

        # 参与度系数 Pi = 1-sum((kis / ki) ** 2)
        for firm in range(num_nodes):
            ki = degree_dict[firm]
            neighbors = set(nx.all_neighbors(g, firm))
            pi = 1
            if ki == 0:
                p_c_array[name, firm] = 0
                continue

            for j in range(len(clusters)):
                kis = len(neighbors & set(clusters[j]))
                pi -= (kis / ki) ** 2
            p_c_array[name, firm] = pi
    # 将参与度系数矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    p_c_df = pd.DataFrame(p_c_array, index=years, columns=list(range(num_firm)))
    p_c_df.to_excel(writer, sheet_name='participation_coefficient')
    print('participation_coefficient 指标计算完成')


# 计算节点所属子群规模
def cluster_size(path, file_name, num_firm, writer):
    # 创建参与系数矩阵（年份×公司数）
    cluster_size_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件读取路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]
        for cluster in clusters:
            size = len(cluster)
            for firm in cluster:
                cluster_size_array[name, firm] = size

    cluster_size_df = pd.DataFrame(cluster_size_array, index=years, columns=list(range(num_firm)))
    cluster_size_df.to_excel(writer, sheet_name='cluster_size')
    print('cluster_size 指标计算完成')


# 计算度中心度的绝对值
def degree_centrality_abs(path, file_name, num_firm, writer):
    degree_centrality_abs_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件读取路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        clusters_edges = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]

        degree_centrality_abs_dict = dict()

        for cluster_edges in clusters_edges:
            g = nx.Graph()
            g.add_edges_from(cluster_edges)
            a = dict(nx.degree(g))
            degree_centrality_abs_dict = {**degree_centrality_abs_dict, **a}

        for key, value in degree_centrality_abs_dict.items():
            degree_centrality_abs_array[name, key] = value

    degree_centrality_abs_df = pd.DataFrame(degree_centrality_abs_array, index=years, columns=list(range(num_firm)))
    degree_centrality_abs_df.to_excel(writer, sheet_name='degree_centrality_abs')
    print('degree_centrality_abs 指标计算完成')


# 计算子群内节点跨簇团的数量
def bridging_cluster(path, file_name, num_firm, writer):
    bridging_cluster_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件读取路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        # 获取数据规模
        num_nodes = len(data['nodes'])
        num_edges = len(data['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(num_edges)]
        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]

        # 创建整个网络完整 Graph
        g = nx.Graph()
        g.add_nodes_from(list(range(num_nodes)))
        g.add_edges_from(edges)
        degree_dict = nx.degree(g)

        for firm in range(num_firm):
            degree_firm = degree_dict[firm]
            neighbors = set(nx.all_neighbors(g, firm))
            num_bridging_cluster = 0
            if degree_firm == 0:
                continue
            for j in range(len(clusters)):
                if (len(neighbors & set(clusters[j]))) > 0:
                    num_bridging_cluster += 1
            bridging_cluster_array[name, firm] = num_bridging_cluster - 1

    # 转换为 DataFrame 格式，并将数据写入 excel 表格中
    bridging_cluster_df = pd.DataFrame(bridging_cluster_array, index=years, columns=list(range(num_firm)))
    bridging_cluster_df.to_excel(writer, sheet_name='跨子群数量')
    print('跨子群数量 指标计算完成')


if __name__ == '__main__':

    # 程序的输入部分
    n = 150                  # 公司总数
    folder_path = './data'   # JSON 文件所在文件夹路径
    file_list = os.listdir(folder_path)

    writer_df = pd.ExcelWriter('./result/static_indicator.xlsx')

    static_indicators(folder_path, file_list, n, writer_df)

    participation_coefficient(folder_path, file_list, n, writer_df)

    cluster_size(folder_path, file_list, n, writer_df)

    degree_centrality_abs(folder_path, file_list, n, writer_df)

    bridging_cluster(folder_path, file_list, n, writer_df)

    writer_df.save()
