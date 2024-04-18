# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# -*- coding: utf-8 -*-
# @Time     :2024/4/17/21:22
# @Auther   :Eat
# @Function :因为手算太麻烦，因此这里自动计算c_1电路的个wire的输出电压
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

from Functions_of_create_basal_graph import generate_input_combinations


def get_signals_labels_for_c1():

    # 可能的输入组合
    inputs_combinations = generate_input_combinations(5)

    # 所有输入情况下的输出label
    out_labels_list = [[-1] * 17 for _ in range(len(inputs_combinations))]

    for i in range(len(inputs_combinations)):

        input_combination_list = list(inputs_combinations[i])

        N1 = input_combination_list[0]
        N2 = input_combination_list[1]
        N3 = input_combination_list[2]
        N6 = input_combination_list[3]
        N7 = input_combination_list[4]
        # print(f'当输入为{N1}, {N3}, {N6}, {N2}, {N7}时')

        N10 = int(not (N1 and N3))
        N11 = int(not (N3 and N6))
        N16 = int(not (N11 and N2))
        N19 = int(not (N11 and N7))
        N22 = int(not (N10 and N16))
        N23 = int(not (N16 and N19))
        # wire_label = [N10, N11, N16, N19, N22, N23]
        # print(f'输出为: {wire_label}\n')

        out_labels_list[i][5] = N22
        out_labels_list[i][6] = N23
        out_labels_list[i][7] = N10
        out_labels_list[i][8] = N11
        out_labels_list[i][9] = N16
        out_labels_list[i][10] = N19

    # print(out_labels_list)
    return out_labels_list
