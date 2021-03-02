# 构建的引用网络是一个有向有权图

import xlrd
import os
import json


def get_graph_inf(graph_path, json_path):
    read_book = xlrd.open_workbook(graph_path)
    read_book_sheet = read_book.sheet_by_index(0)
    rows = read_book_sheet.nrows
    cols = read_book_sheet.ncols
    # 获取唯一的id
    source = read_book_sheet.col_values(0)[1:]
    target = read_book_sheet.row_values(0)[1:]
    nodes_list = sorted(list(set(source + target)))

    nodes_dict = []
    for i in range(len(nodes_list)):
        nodes_dict.append({'name': nodes_list[i]})
    # 获取link的信息
    links_temper = []
    links_dict = []
    for i in range(1, rows):
        for j in range(1, cols):
            value = read_book_sheet.cell_value(i, j)
            if value == '' or value == 0.0 or source[i - 1] == target[j - 1]:
                continue
            else:
                link = sorted([source[i - 1], target[j - 1]])
                if link not in links_temper:
                    links_temper.append(link)
                    links_dict.append({'source': nodes_list.index(source[i - 1]),
                                       'target': nodes_list.index(target[j - 1]),
                                       'value': int(value)})
                else:
                    links_dict[links_temper.index(link)]['value'] += int(value)

    graph_inf = {'nodes': nodes_dict, 'links': links_dict}
    json.dump(graph_inf, open(json_path, 'w', encoding='UTF-8'))


if __name__ == '__main__':
    # 修改处
    excel_name = '引用网络_示例.xlsx'

    excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '0.数据\\输入数据', excel_name)
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '0.数据\\json',
                             os.path.splitext(excel_name)[0] + '.json')
    get_graph_inf(excel_path, json_path)
