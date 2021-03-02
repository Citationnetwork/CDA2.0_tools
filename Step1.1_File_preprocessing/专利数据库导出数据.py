# 构建的合作关系网络是一个无向有权图

import json
import os


def get_graph_inf(graph_path, json_path):
    txt_file = open(graph_path, 'r', encoding='UTF-16')
    nodes_list = []
    links_list = []
    for each_line in txt_file:
        if each_line[:4] == 'pad:':
            nodes_temper = each_line[4:-1].replace(' ', '').split('|')
            nodes_list += nodes_temper
            for i in range(len(nodes_temper)):
                for j in range(i + 1, len(nodes_temper)):
                    links_list.append([nodes_temper[i], nodes_temper[j]])
    # 写入字典
    nodes_list = sorted(list(set(nodes_list)))
    nodes_dict = []
    for i in range(len(nodes_list)):
        nodes_dict.append({'name': nodes_list[i],
                           'group': 1})
    links_dict = []
    links_temper = []
    for i in range(len(links_list)):
        if links_list[i] not in links_temper:
            links_temper.append(links_list[i])
            links_dict.append({'source': nodes_list.index(links_list[i][0]),
                               'target': nodes_list.index(links_list[i][1]),
                               'value': 1})
        else:
            links_dict[links_temper.index(links_list[i])]['value'] += 1

    graph_inf = {'nodes': nodes_dict, 'links': links_dict}
    json.dump(graph_inf, open(json_path, 'w', encoding='UTF-8'))


if __name__ == '__main__':
    # 修改处
    txt_name = '专利数据库导出数据_示例.txt'

    txt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\data_输入', txt_name)
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\result_输出',
                             os.path.splitext(txt_name)[0] + '.json')
    get_graph_inf(txt_path, json_path)
    print('数据预处理完成！')


