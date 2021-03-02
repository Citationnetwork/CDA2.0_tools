#!/usr/bin/env python 3.7
# -*- coding: utf-8 -*-
"""
# @version : V1.0
# @Author  : mxz
# @contact : 925813094@qq.com
# @Time    : 2020.3.2
# @File    : index.py
# @Software: PyCharm
"""
# 计算子群动态指标指标
# JSON 文件需按年份有序命名，文件名的前四位必须为年份，确保程序能正常处理文件
# 文件路径为该程序路径下的data文件夹

import os
import math
import json
import numpy as np
import pandas as pd
import networkx as nx


# 处理多年份数据的 z_score ，并整合为 DataFrame 格数输出为excel表格
def z_score(path, file_name, num_firm, writer):
    # 创建 z_score 矩阵（年份×公司数）
    z_score_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件路径
        file_path = path + '/' + file_name[name]

        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        clusters = [data['clusters'][k]['cluster'] for k in range(len(data['clusters']))]
        clusters_edges = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]

        # 簇团数量
        num_clusters = len(clusters)
        # 创建一个Graph
        g = nx.Graph()
        # 创建 z_score 字典
        z_score_dict = {}
        for i in range(num_clusters):
            # 如果簇团内只有一家公司，则其 z_score 设为1
            if len(clusters[i]) == 1:
                z_score_dict[clusters[i][0]] = 0
            # 如果簇团内公司大于一家，按照公式计算 z_score
            else:
                g.clear()
                g.add_nodes_from(clusters[i])
                g.add_edges_from(clusters_edges[i])
                degree = dict(g.degree())
                degree_mean = np.mean(list(degree.values()))
                degree_std = np.std(list(degree.values()))
                for key, value in degree.items():
                    if degree_std == 0:
                        z_score_dict[key] = 1
                    else:
                        z_score_dict[key] = (value - degree_mean) / degree_std
        for i in range(num_firm):
            z_score_array[name, i] = z_score_dict[i]

    # 将 z_score 矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    z_score_df = pd.DataFrame(z_score_array, index=years, columns=list(range(num_firm)))
    z_score_df.to_excel(writer, sheet_name='z_score')
    print('z_score 指标计算完成')


# Current bridging ties x Current local ties （进⾏中⼼化和对数处理后，相乘），可批量化处理文件
def bridging_multiplying_local(path, file_name, num_firm, writer):
    # 创建 Current bridging ties x Current local ties 矩阵（年份×公司数）
    b_l_array = np.zeros((len(file_name), num_firm))
    years = []
    for name in range(len(file_name)):
        years.append(file_name[name][0:4])
        # 文件路径
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
        clusters_edges = [data['clusters'][k]['edges'] for k in range(len(data['clusters']))]

        # 创建完全网络的 Graph
        g = nx.Graph()
        g.add_nodes_from(list(range(num_firm)))
        g.add_edges_from(edges)
        # 获取节点的度，转换为字典格式
        degree = dict(g.degree())
        # 簇团内节点的度字典
        degree_local = {}
        # 簇团内的某点的与簇团外的所有点的度字典
        degree_bridging = {}

        # 计算 degree_local
        for j in range(len(clusters)):
            g_cluster = nx.Graph()
            g_cluster.add_nodes_from(clusters[j])
            g_cluster.add_edges_from(clusters_edges[j])
            degree_local_j = dict(g_cluster.degree())
            degree_local = {**degree_local, **degree_local_j}

        # 计算 degree_bridging
        for firm in range(num_nodes):
            degree_bridging[firm] = degree[firm] - degree_local[firm]

        # 求均值
        degree_local_mean = np.mean(list(degree_local.values()))
        degree_bridging_mean = np.mean(list(degree_bridging.values()))

        # 计算 Current bridging ties x Current local ties 矩阵（年份×公司数）
        for firm in range(num_nodes):
            a = abs(degree_bridging[firm]-degree_bridging_mean)
            b = abs(degree_local[firm] - degree_local_mean)
            if a == 0 or b == 0:
                b_l_array[name, firm] = 0
            else:
                b_l = math.log(a, 10) * math.log(b, 10)
                b_l_array[name, firm] = b_l

    # 将 Current bridging ties x Current local ties 矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    b_l_df = pd.DataFrame(b_l_array, index=years, columns=list(range(num_firm)))
    b_l_df.to_excel(writer, sheet_name='bridging_multiplying_local')
    print('bridging_multiplying_local 指标计算完成')


# new_neighbor_bridging_ties 指标计算函数，可批量化处理文件
def new_neighbor_bridging_ties(path, file_name, num_firm, writer):
    # 创建 new_bridging_ties 矩阵（（年份-1)×公司数），第一年没有
    b_t_array = np.zeros((len(file_name)-1, num_firm))
    years = []

    for name in range(len(file_name)-1):
        years.append(file_name[name+1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())

        # 获取数据规模
        num_nodes_1 = len(data_1['nodes'])
        num_edges_1 = len(data_1['links'])
        num_nodes_2 = len(data_2['nodes'])
        num_edges_2 = len(data_2['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges_1 = [(data_1['links'][k]['source'], data_1['links'][k]['target']) for k in range(num_edges_1)]
        edges_2 = [(data_2['links'][k]['source'], data_2['links'][k]['target']) for k in range(num_edges_2)]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]
        clusters_edges_2 = [data_2['clusters'][k]['edges'] for k in range(len(data_2['clusters']))]

        g_t1 = nx.Graph()
        g_t1.add_nodes_from(list(range(num_nodes_1)))
        g_t1.add_edges_from(edges_1)
        g_t2 = nx.Graph()
        g_t2.add_nodes_from(list(range(num_nodes_2)))
        g_t2.add_edges_from(edges_2)

        for module in range(len(clusters_2)):
            for firm in clusters_2[module]:
                firm_neighbors = list(nx.all_neighbors(g_t2, firm))
                neighbors_edges_t1 = []
                neighbors_edges_t2 = []
                for firm_neighbor in firm_neighbors:
                    neighbors_edges_t1 += nx.edges(g_t1, firm_neighbor)
                    neighbors_edges_t2 += nx.edges(g_t2, firm_neighbor)
                neighbors_edges_t1_sorted, neighbors_edges_t2_sorted = set(), set()
                for edge in neighbors_edges_t1:
                    edge = list(edge)
                    edge.sort()
                    edge = tuple(edge)
                    neighbors_edges_t1_sorted.add(edge)
                for edge in neighbors_edges_t2:
                    edge = list(edge)
                    edge.sort()
                    edge = tuple(edge)
                    neighbors_edges_t2_sorted.add(edge)
                new_edges = set(neighbors_edges_t2_sorted) - set(neighbors_edges_t1_sorted)
                clusters_edges_sorted = set()
                for edge in clusters_edges_2[module]:
                    edge = tuple(edge)
                    clusters_edges_sorted.add(edge)
                new_edges_bridging = new_edges - clusters_edges_sorted
                d_j_beta = len(new_edges_bridging)
                if d_j_beta == 0:
                    b_t_array[name, firm] = 0
                else:
                    b_t_array[name, firm] = math.log(d_j_beta, 10)

    # 将 bridging_ties 矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    b_t_df = pd.DataFrame(b_t_array, index=years, columns=list(range(num_firm)))
    b_t_df.to_excel(writer, sheet_name='new_neighbor_bridging_ties')
    print('new_neighbor_bridging_ties 指标计算完成')


# potential_bridging_partners 指标计算，可批量化处理文件
def potential_bridging_partners(path, file_name, num_firm, writer):
    # 创建 bridging_ties 矩阵（（年份-1)×公司数），第一年没有
    pbp_array = np.zeros((len(file_name) - 1, num_firm))
    years = []

    for name in range(len(file_name) - 1):
        years.append(file_name[name + 1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name + 1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())

        # 获取数据规模
        num_nodes_1 = len(data_1['nodes'])
        num_edges_1 = len(data_1['links'])
        num_nodes_2 = len(data_2['nodes'])
        num_edges_2 = len(data_2['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges_1 = [(data_1['links'][k]['source'], data_1['links'][k]['target']) for k in range(num_edges_1)]
        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_edges_1 = [data_1['clusters'][k]['edges'] for k in range(len(data_1['clusters']))]
        edges_2 = [(data_2['links'][k]['source'], data_2['links'][k]['target']) for k in range(num_edges_2)]

        g_t1 = nx.Graph()
        g_t1.add_nodes_from(list(range(num_nodes_1)))
        g_t1.add_edges_from(edges_1)
        g_t2 = nx.Graph()
        g_t2.add_nodes_from(list(range(num_nodes_2)))
        g_t2.add_edges_from(edges_2)

        for module in range(len(clusters_1)):
            n_c = set()
            c = set()
            inside_firms = set(clusters_1[module])
            outside_firms = set(range(150)) - inside_firms
            g_cluster = nx.Graph()
            g_cluster.add_nodes_from(clusters_1[module])
            g_cluster.add_edges_from(clusters_edges_1[module])
            for firm in inside_firms:
                firm_neighbors_t1 = set(nx.all_neighbors(g_t1, firm))
                firm_neighbors_t2 = set(nx.all_neighbors(g_t2, firm))
                new_neighbors = firm_neighbors_t2 - firm_neighbors_t1
                if len(new_neighbors) > 0:
                    c.add(firm)
            for firm in outside_firms:
                firm_neighbors_t1 = set(nx.all_neighbors(g_t1, firm))
                firm_neighbors_t2 = set(nx.all_neighbors(g_t2, firm))
                new_neighbors = firm_neighbors_t2 - firm_neighbors_t1
                if len(new_neighbors) > 0:
                    n_c.add(firm)
            for firm in clusters_1[module]:
                firm_neighbors = set(nx.all_neighbors(g_t1, firm))
                n_c_d = n_c - firm_neighbors
                c_d = c - firm_neighbors
                if len(n_c_d) == 0 or len(c_d) == 0:
                    pbp_array[name, firm] = 0
                else:
                    pbp = math.log(len(n_c_d) / len(c_d), 10)
                    pbp_array[name, firm] = pbp

    # 将 potential_bridging_partners 矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    pbp_df = pd.DataFrame(pbp_array, index=years, columns=list(range(num_firm)))
    pbp_df.to_excel(writer, sheet_name='potential_bridging_partners')
    print('potential_bridging_partners 指标计算完成')


# new_bridging_ties指标计算
def new_bridging_ties(path, file_name, num_firm, writer):
    # 创建 new_bridging_ties 矩阵（（年份-1)×公司数），第一年没有
    nbt_array = np.zeros((len(file_name)-1, num_firm))
    years = []

    for name in range(len(file_name)-1):
        years.append(file_name[name+1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())

        # 获取数据规模
        num_nodes_1 = len(data_1['nodes'])
        num_edges_1 = len(data_1['links'])
        num_nodes_2 = len(data_2['nodes'])
        num_edges_2 = len(data_2['links'])

        # 读取簇团信息及簇团内连接信息  连接及权重信息
        edges_1 = [(data_1['links'][k]['source'], data_1['links'][k]['target']) for k in range(num_edges_1)]
        edges_2 = [(data_2['links'][k]['source'], data_2['links'][k]['target']) for k in range(num_edges_2)]

        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]

        clusters_edges_1 = [data_1['clusters'][k]['edges'] for k in range(len(data_1['clusters']))]
        clusters_edges_2 = [data_2['clusters'][k]['edges'] for k in range(len(data_2['clusters']))]

        edges_bridging_1, edges_bridging_2 = set(edges_1), set(edges_2)

        for cluster in clusters_edges_1:
            for i in range(len(cluster)):
                cluster[i] = tuple(cluster[i])
            edges_bridging_1 -= set(cluster)

        for cluster in clusters_edges_2:
            for i in range(len(cluster)):
                cluster[i] = tuple(cluster[i])
            edges_bridging_2 -= set(cluster)

        new_bridging = list(edges_bridging_2 - edges_bridging_1)

        g = nx.Graph()
        g.add_edges_from(new_bridging)

        degree_new_bridging = dict(nx.degree(g))

        for key, value in degree_new_bridging.items():
            nbt_array[name, key] = value

    # 将 bridging_ties 矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    nbt_df = pd.DataFrame(nbt_array, index=years, columns=list(range(num_firm)))
    nbt_df.to_excel(writer, sheet_name='new_bridging_ties')
    print('new_bridging_ties 指标计算完成')


# t年之前该公司所属的不同社区的数量，但不包括当前社区。如果公司之前没社区⾪属关系，则此变量设置为0
def prior_community_affiliations(path, file_name, num_firm, writer):
    pca_array = np.ones((len(file_name)-1, num_firm))
    years = []
    for name in range(len(file_name)-1):
        years.append(file_name[name+1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())
        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]

        # 计算重叠率矩阵
        for i in range(len(clusters_2)):
            for j in range(len(clusters_1)):
                if len(clusters_1[j]) == 1:
                    pca_array[name, clusters_1[j][0]] = 0
                else:
                    a = len(set(clusters_1[j]) & set(clusters_2[i]))
                    b = len(set(clusters_1[j]) | set(clusters_2[i]))
                    overlap_rate_value = float(a / b)
                    if overlap_rate_value > 0.3:
                        for firm in set(clusters_1[j]) & set(clusters_2[i]):
                            pca_array[name, firm] = 0

    # 将重叠率矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    pca_df = pd.DataFrame(pca_array, index=years, columns=list(range(num_firm)))
    pca_df.to_excel(writer, sheet_name='prior_community_affiliations')
    print('prior_community_affiliations 指标计算完成')


# 批量文件处理，计算相邻年份簇团的重叠率
def overlap_rate(path, file_name):
    writer = pd.ExcelWriter('./result/overlap_rate.xlsx')
    for name in range(len(file_name)-1):
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())
        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]

        # 创建重叠率矩阵（t年簇团数 × t+1年簇团数）
        overlap_rate_array = np.zeros((len(clusters_1), len(clusters_2)))

        # 计算重叠率矩阵
        for i in range(len(clusters_1)):
            for j in range(len(clusters_2)):
                a = len(set(clusters_1[i]) & set(clusters_2[j]))
                b = len(set(clusters_1[i]) | set(clusters_2[j]))
                overlap_rate_c = float(a / b)
                overlap_rate_array[i, j] = overlap_rate_c

        # 将重叠率矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
        overlap_rate_df = pd.DataFrame(overlap_rate_array,
                                       index=list(range(len(clusters_1))),
                                       columns=list(range(len(clusters_2))))
        sheet = file_name[name][0:4]+'_'+file_name[name+1][0:4]+'_overlap_rate'
        overlap_rate_df.to_excel(writer, sheet_name=sheet)

    writer.save()
    print('overlap_rate 指标计算完成')


# 批量文件处理，计算每个节点的重叠率与更迭率
def overlap_rate_firm_1(path, file_name, num_firm, writer):
    overlap_rate_array = np.zeros((len(file_name)-1, num_firm))
    turn_over_array = np.zeros((len(file_name)-1, num_firm))
    years = []
    for name in range(len(file_name)-1):
        years.append(file_name[name+1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())
        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]

        # 计算重叠率矩阵
        for i in range(len(clusters_2)):
            overlap_rate_c = []
            for j in range(len(clusters_1)):
                a = len(set(clusters_1[j]) & set(clusters_2[i]))
                b = len(set(clusters_1[j]) | set(clusters_2[i]))
                overlap_rate_c.append(float(a / b))
            overlap_rate_value = max(overlap_rate_c)
            turn_over_value = 1 - overlap_rate_value
            for firm in clusters_2[i]:
                overlap_rate_array[name, firm] = overlap_rate_value
                turn_over_array[name, firm] = turn_over_value

    # 将重叠率矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    overlap_rate_df = pd.DataFrame(overlap_rate_array, index=years, columns=list(range(num_firm)))
    overlap_rate_df.to_excel(writer, sheet_name='overlap_rate_1')
    turn_over_df = pd.DataFrame(turn_over_array, index=years, columns=list(range(num_firm)))
    turn_over_df.to_excel(writer, sheet_name='turn_over_1')
    print('overlap_rate 与 turn_over 指标计算完成')


# 批量文件处理，计算每个节点的重叠率与更迭率
def overlap_rate_firm_2(path, file_name, num_firm, writer):
    overlap_rate_array = np.zeros((len(file_name)-1, num_firm))
    turn_over_array = np.zeros((len(file_name)-1, num_firm))
    years = []
    for name in range(len(file_name)-1):
        years.append(file_name[name+1][0:4])
        # 两个连续年份文件路径
        file_path_1 = path + '/' + file_name[name]
        file_path_2 = path + '/' + file_name[name+1]
        # 读取数据
        with open(file_path_1, 'r', encoding='utf-8') as f:
            data_1 = json.loads(f.read())
        with open(file_path_2, 'r', encoding='utf-8') as f:
            data_2 = json.loads(f.read())
        clusters_1 = [data_1['clusters'][k]['cluster'] for k in range(len(data_1['clusters']))]
        clusters_2 = [data_2['clusters'][k]['cluster'] for k in range(len(data_2['clusters']))]

        # 计算重叠率矩阵
        for i in range(num_firm):
            for j in range(len(clusters_1)):
                if i not in clusters_1[j]:
                    continue
                for k in range(len(clusters_2)):
                    if i not in clusters_1[k]:
                        continue
                    a = len(set(clusters_1[j]) & set(clusters_2[k]))
                    b = len(set(clusters_1[j]) | set(clusters_2[k]))
                    overlap_rate_value = float(a / b)
                    turn_over_value = 1 - overlap_rate_value
                    overlap_rate_array[name, i] = overlap_rate_value
                    turn_over_array[name, i] = turn_over_value
                    break
                break

    # 将重叠率矩阵转换为 DataFrame 格式，并将数据写入 excel 表格中
    overlap_rate_df = pd.DataFrame(overlap_rate_array, index=years, columns=list(range(num_firm)))
    overlap_rate_df.to_excel(writer, sheet_name='overlap_rate_2')
    turn_over_df = pd.DataFrame(turn_over_array, index=years, columns=list(range(num_firm)))
    turn_over_df.to_excel(writer, sheet_name='turn_over_2')
    print('overlap_rate 与 turn_over 指标计算完成')


if __name__ == '__main__':

    # 函数的输入部分
    n = 150                  # 公司总数
    folder_path = './data'   # JSON 文件所在文件夹路径
    file_list = os.listdir(folder_path)

    writer_df = pd.ExcelWriter('./result/dynamic_indicator.xlsx')

    prior_community_affiliations(folder_path, file_list, n, writer_df)

    z_score(folder_path, file_list, n, writer_df)

    bridging_multiplying_local(folder_path, file_list, n, writer_df)

    new_neighbor_bridging_ties(folder_path, file_list, n, writer_df)

    potential_bridging_partners(folder_path, file_list, n, writer_df)

    new_bridging_ties(folder_path, file_list, n, writer_df)

    overlap_rate(folder_path, file_list)

    overlap_rate_firm_1(folder_path, file_list, n, writer_df)

    overlap_rate_firm_2(folder_path, file_list, n, writer_df)

    writer_df.save()

