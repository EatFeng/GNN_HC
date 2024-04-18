# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# -*- coding: utf-8 -*-
# @Time     :2024/4/16/11:27
# @Auther   :Eat
# @Function : 电源数据集类，其中包含在所有输入下的电路图
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

import torch
from create_labels_for_c_full_adder import get_signals_labels_for_full_adder


def graphs_combined_with_signals(basal_graph, input_combinations):

    graphs_list = []

    # 这里的label只是针对半加器设置
    # signal_labels = [[-1, -1, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
    #                  [-1, -1, 0, 1, 1, 1, 0, -1, -1, -1, -1, -1],
    #                  [-1, -1, 0, 1, 1, 0, 1, -1, -1, -1, -1, -1],
    #                  [-1, -1, 1, 0, 0, 1, 1, -1, -1, -1, -1, -1]]
    # 这里的label只是针对C_1电路
    signal_labels = get_signals_labels_for_full_adder()
    # print(signal_labels)

    for inputs in input_combinations:
        # 克隆基图，以防更改原始数据
        graph = basal_graph.clone()

        # 更新输入节点特征
        for i, node_feature in enumerate(graph.x):
            if node_feature[4] == 1:  # 第5位是输入节点的标识
                high_low = int(inputs[i])  # 从输入组合中获取当前节点的高低状态
                # 更新对应的高低特征位
                node_feature[6] = 1 - high_low  # 第7位是低电平标识
                node_feature[7] = high_low  # 假设第8位是高电平标识

        graphs_list.append(graph)

    label_index = 0

    for graph in graphs_list:
        graph.y = torch.tensor(signal_labels[label_index])
        label_index += 1

    # print(graphs_list)
    # print(graphs_list[0].x)

    return graphs_list
