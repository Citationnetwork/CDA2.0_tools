# 构建的共现网络是一个无向向有权图

import xlrd
import os
import json


def get_graph_inf(graph_path, json_path):
    read_book = xlrd.open_workbook(graph_path)
    read_book_sheet = read_book.sheet_by_index(0)
    rows = read_book_sheet.nrows
    # 获取唯一的id
    nodes_list = read_book_sheet.row_values(0)[1:]

    nodes_dict = []
    for i in range(len(nodes_list)):
        nodes_dict.append({'name': nodes_list[i],
                           'group': 1})
    # 获取link的信息
    links_dict = []
    for i in range(1, rows):
        for j in range(i + 1, rows):
            value = read_book_sheet.cell_value(i, j)
            if value == '' or value == 0.0:
                continue
            else:
                links_dict.append({'source': nodes_list.index(nodes_list[i - 1]),
                                   'target': nodes_list.index(nodes_list[j - 1]),
                                   'value': int(value)})

    graph_inf = {'nodes': nodes_dict, 'links': links_dict}
    json.dump(graph_inf, open(json_path, 'w', encoding='UTF-8'))


if __name__ == '__main__':
    # 修改处
    excel_name = '共现网络_示例.xlsx'

    excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\data_输入', excel_name)
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Step1.1_文件预处理\\data_输入',
                             os.path.splitext(excel_name)[0] + '.json')
    get_graph_inf(excel_path, json_path)
