# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# -*- coding: utf-8 -*-
# @Time     :2024/4/16/10:53
# @Auther   :Eat
# @Function :这里存放着用于构建基图的函数
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

import itertools
import re
import torch
from torch_geometric.data import Data
from collections import deque, defaultdict


def netlist_to_circuit_components(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # 处理声明可能跨多行的情况
        content = content.replace('\n', ' ')

        # 定义正则表达式
        input_output_pattern = re.compile(r'(input|output)\s+([\w,\[\] ]+);')
        logic_gate_pattern = re.compile(r'(nand|nor|not|and|or|xor)\s+(\w+)\s*\(([\w,\s]+)\);')
        wire_pattern = re.compile(r'wire\s+([\w,\s]+);')

        # 解析wire声明
        wires = wire_pattern.findall(content)
        wires = [wire.strip() for wire in ','.join(wires).replace(' ', '').split(',') if wire]

        # 解析输入输出声明
        inputs = []
        outputs = []
        for io_type, names in input_output_pattern.findall(content):
            name_list = [name.strip() for name in names.split(',')]
            if io_type == 'input':
                inputs.extend(name_list)
            elif io_type == 'output':
                outputs.extend(name_list)

        # 解析逻辑门实例化
        gates = [{'type': gate[0], 'name': gate[1], 'connections': [conn.strip() for conn in gate[2].split(',')]}
                 for gate in logic_gate_pattern.findall(content)]

        return gates, wires, inputs, outputs

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return [], {}, []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], {}, []


def generate_input_combinations(num_inputs):
    # 生成所有可能的输入信号组合（二进制形式，1表示高，0表示低）
    return list(itertools.product([0, 1], repeat=num_inputs))


def create_graph_base(gates, wires, inputs, outputs):
    node_index = {}
    x = []
    edge_index = [[], []]
    current_index = 0

    # 节点特征向量模板
    feature_template = {
        'nand': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'not':  [0, 1, 0, 0, 0, 0, 0, 0, 0],
        'nor':  [0, 0, 1, 0, 0, 0, 0, 0, 0],
        'wire': [0, 0, 0, 1, 0, 0, 0, 0, 0],
        'in':   [0, 0, 0, 0, 1, 0, 0, 0, 0],
        'out':  [0, 0, 0, 0, 0, 1, 0, 0, 0]
    }

    # 初始化输入节点
    for input_name in inputs:
        node_index[input_name] = current_index
        x.append(feature_template['in'])
        current_index += 1

    # 初始化输出节点
    for output_name in outputs:
        node_index[output_name] = current_index
        x.append(feature_template['out'])
        current_index += 1

    # 初始化wire节点
    for wire_name in wires:
        node_index[wire_name] = current_index
        x.append(feature_template['wire'])
        current_index += 1

    # 初始化门节点和设置边
    for gate in gates:
        if gate['name'] not in node_index:
            node_index[gate['name']] = current_index
            x.append(feature_template[gate['type']])
            current_index += 1

        output_port = gate['connections'][0]
        input_ports = gate['connections'][1:]
        src = node_index[gate['name']]

        # 逻辑门到输出
        if output_port in node_index:
            edge_index[0].append(src)
            edge_index[1].append(node_index[output_port])

        # 输入到逻辑门
        for input_port in input_ports:
            if input_port in node_index:
                edge_index[0].append(node_index[input_port])
                edge_index[1].append(src)

    # 创建图对象
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    train_mask = torch.zeros(len(x), dtype=torch.bool)  # 初始化全部为False
    labels = torch.full((len(x),), -1, dtype=torch.long)  # 初始标签为-1

    return Data(x=x, edge_index=edge_index, train_mask=train_mask, y=labels)


def get_node_depth(data):
    num_nodes = data.num_nodes
    depth = [-float('inf')] * num_nodes  # 初始化深度为负无穷
    in_degree = [0] * num_nodes  # 存储每个节点的入度

    # 计算每个节点的入度
    for source, target in data.edge_index.t():
        in_degree[target.item()] += 1

    # 找到所有入度为0的节点（输入节点）
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    for node in queue:
        depth[node] = 0  # 输入节点深度设置为0

    # 按拓扑顺序处理每个节点
    while queue:
        node = queue.popleft()
        current_depth = depth[node]
        for neighbor in data.edge_index[1][data.edge_index[0] == node]:
            in_degree[neighbor.item()] -= 1
            # 更新深度
            depth[neighbor.item()] = max(depth[neighbor.item()], current_depth + 1)
            if in_degree[neighbor.item()] == 0:
                queue.append(neighbor.item())

    # 归一化深度
    max_depth = max(depth)
    min_depth = min(d for d in depth if d != -1)
    normalized_depth = [(d - min_depth) / (max_depth - min_depth) if d != -1 else 0 for d in depth]

    # 更新节点特征
    for i in range(num_nodes):
        data.x[i, 8] = normalized_depth[i]  # 假设深度存放在第9位

    # 分组相同深度的节点
    depth_groups = defaultdict(list)
    for i in range(num_nodes):
        if data.x[i][5] == 1 or data.x[i][3] == 1:  # 假设wire和output的标识位于第三和第五位
            depth_groups[depth[i]].append(i)

    # 将分组转换为列表并按深度排序
    sorted_depth_groups = [depth_groups[d] for d in sorted(depth_groups)]

    return data, sorted_depth_groups
