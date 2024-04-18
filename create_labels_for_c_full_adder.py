# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# -*- coding: utf-8 -*-
# @Time     :2024/4/18/11:54
# @Auther   :Eat
# @Function :因为手算太麻烦，因此这里自动计算c_fulladder电路的个wire的输出电压
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

from Functions_of_create_basal_graph import generate_input_combinations


def get_signals_labels_for_full_adder():

    # 可能的输入组合
    inputs_combinations = generate_input_combinations(3)

    # 所有输入情况下的输出label
    out_labels_list = [[-1] * 27 for _ in range(len(inputs_combinations))]

    for i in range(len(inputs_combinations)):

        input_combination_list = list(inputs_combinations[i])

        Cin = input_combination_list[0]
        A = input_combination_list[1]
        B = input_combination_list[2]
        # print(f'当输入为Cin:{Cin}, A:{A}, B:{B}时')

        N1 = int(not (A and B))
        N2 = int(not (A and N1))
        N3 = int(not (N1 and B))
        N10 = int(not N1)
        N4 = int(not (N2 and N3))
        N5 = int(not (N4 and Cin))
        N6 = int(not (Cin and N5))
        N8 = int(not (N4 and N5))
        Sum = int(not (N6 and N8))
        N9 = int(not N5)
        N11 = int(not (N9 or N10))
        Carry = int(not N11)
        # print(f'输出应当为: N1={N1}, N2={N2}, N3={N3}, N10={N10}, N4={N4}, N5={N5}, N6={N6}, N8={N8}, N9={N9}, N11={N11}'
        #       f', Sum={Sum}, Carry={Carry}\n')

        out_labels_list[i][3] = Carry
        out_labels_list[i][4] = Sum
        out_labels_list[i][5] = N1
        out_labels_list[i][6] = N2
        out_labels_list[i][7] = N3
        out_labels_list[i][8] = N4
        out_labels_list[i][9] = N5
        out_labels_list[i][10] = N6
        out_labels_list[i][11] = N8
        out_labels_list[i][12] = N9
        out_labels_list[i][13] = N10
        out_labels_list[i][14] = N11

    # print(out_labels_list)
    return out_labels_list
