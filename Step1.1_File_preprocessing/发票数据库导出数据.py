# 构建的合作关系网络是一个有向有权图

import xlrd
import os
import json


def get_graph_inf(graph_path, json_path):
    read_book = xlrd.open_workbook(graph_path)
    read_book_sheet = read_book.sheet_by_index(0)
    rows = read_book_sheet.nrows
    # 获取唯一的id
    source = read_book_sheet.col_values(0)[1:]
    target = read_book_sheet.col_values(1)[1:]
    nodes_list = sorted(list(set(source + target)))
    links_list = list(zip(source, target))

    nodes_dict = []
    for i in range(len(nodes_list)):
        nodes_dict.append({'name': nodes_list[i],
                           'group': 1})
    # 获取link的信息
    links_dict = []
    links_temper = []
    for i in range(len(links_list)):
        if links_list[i] not in links_temper:
            links_temper.append(links_list[i])
            links_dict.append({'source': nodes_list.index(source[i]),
                               'target': nodes_list.index(target[i]),
                               'value': 1})
        else:
            links_dict[links_temper.index(links_list[i])]['value'] +=1

    graph_inf = {'nodes': nodes_dict, 'links': links_dict}
    json.dump(graph_inf, open(json_path, 'w', encoding='UTF-8'))


if __name__ == '__main__':
    # 修改处
    excel_name = '发票数据库导出数据_示例.xlsx'

    excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\data_输入', excel_name)
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\data_输入',
                             os.path.splitext(excel_name)[0] + '.json')
    get_graph_inf(excel_path, json_path)
