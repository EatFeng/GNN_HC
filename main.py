from Functions_of_create_basal_graph import netlist_to_circuit_components
from Functions_of_create_basal_graph import create_graph_base
from Functions_of_create_basal_graph import generate_input_combinations
from Functions_of_create_basal_graph import get_node_depth
from CircuitDataset import graphs_combined_with_signals
from Model import GCN, train, evaluate, predict
import torch
import copy


def update_train_mask(data_list, active_indices):
    for data in data_list:
        data.train_mask.fill_(False)  # 首先将所有的train_mask设置为False
        data.train_mask[active_indices] = True  # 只有指定索引的节点的train_mask设置为True
        # print(data.train_mask)

    return data_list


def update_node_features(data, idx, feature):
    data.x[idx] = torch.tensor(feature, dtype=torch.float)
    # print(data.x)


def get_number_tuple(num_inputs):

    input_signals_tuple = []

    for i in range(num_inputs):
        # 循环获得有效输入
        user_input = input(f'请输入第{i}个输入端口的输入信号: ')
        try:
            if user_input == '0' or user_input == '1':
                input_signals_tuple.extend(user_input)
            else:
                print('无效输入')
        except ValueError:
            print('无效输入')

    # 将列表转为元组
    input_signals_list = tuple(input_signals_tuple)

    return [input_signals_list]


if __name__ == '__main__':

    # 网表文件路径
    file_path = 'netlist/c_fulladder.v'

    # 从网表中解析components的信息
    gates, wires, inputs, outputs = netlist_to_circuit_components(file_path)
    # print(gates, wires, inputs, outputs)
    # print(len(inputs))

    # 根据components的信息构建基图
    basal_graph = create_graph_base(gates, wires, inputs, outputs)
    # print(basal_graph)
    # x = [nand, not, wire, in, out, in_low, in_high, depth]
    # print(basal_graph.x)
    # print(basal_graph.edge_index)

    # 计算图中节点的深度信息
    # depth_groups中存放着基图从小到大的深度相同的wire或out节点的索引
    basal_graph_with_depth, depth_groups = get_node_depth(basal_graph)
    print(depth_groups)
    # print(basal_graph_with_depth.x)

    # 生成所有可能的输入信号的组合
    input_combinations = generate_input_combinations(len(inputs))
    # for combination in input_combinations:
    #     print(combination)

    # root = 'G:/Python/GNN_HC/dataset'
    graphs_list = graphs_combined_with_signals(basal_graph_with_depth, input_combinations)

    # 确认train_mask和y的类型
    # print(graphs_list.data.train_mask.dtype)  # 应该输出 torch.bool
    # print(graphs_list.data.y.dtype)  # 应该输出适用于分类任务的数据类型，如 torch.long

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=9, num_classes=2).to(device)

    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 200
    best_model_path = 'G:/Python/GNN_HC/models/best_model_for_c_full_adder.pth'
    patience = 30

    # ---------- train ---------- #
    train_enable = False

    if train_enable:
        for depth_level, indices in enumerate(depth_groups):
            graphs_list = update_train_mask(graphs_list, indices)
            # print(graphs_list[depth_level].train_mask)
            # print(graphs_list[depth_level].x)
            # 充值早停计数器
            patience_counter = 0
            min_loss = 10.0

            # 在训练下个深度前 载入之前训练好的模型
            best_model_wts = torch.load(best_model_path)
            model.load_state_dict(best_model_wts)

            for epoch in range(epochs):  # 早停条件可以根据需要设置
                train_loss = train(model, graphs_list, optimizer, criterion, device)
                val_acc = evaluate(model, graphs_list, device)
                print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {val_acc:.4f}')
                if val_acc == 1.0:
                    if min_loss >= train_loss:
                        min_loss = train_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, best_model_path)
                        print(f'New best model saved at epoch {epoch} in depth{depth_level} '
                              f'with accuracy: {min_loss:.4f}')
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= patience:
                    print(f"Reached 100% accuracy with the lowest loss; "
                          f"stopping training at this depth level {depth_level}.\n")
                    break

            # 预测当前级别的wire节点输出
            for graph in graphs_list:
                predictions = predict(model, graph, device)
                # 更新节点特征
                for wire_index in depth_groups[depth_level]:
                    if predictions[wire_index]:
                        new_feature = [0, 0, 0, 1, 0, 0, 0, 1, 0]
                    else:
                        new_feature = [0, 0, 0, 1, 0, 0, 1, 0, 0]

                    update_node_features(graph, wire_index, new_feature)

        # 保存训练好的模型
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, best_model_path)
        print(f'Trained model saved at {best_model_path}\n')

    # ---------- simulate ---------- #
    simulate_enable = True

    if simulate_enable:

        # 在仿真前，载入最佳模型权重
        best_model_wts = torch.load(best_model_path)
        model.load_state_dict(best_model_wts)

        # 首先将输入信号特征加入到基图中
        print(f'请输入{file_path}电路的{len(inputs)}个输入信号: \n')
        input_signals = get_number_tuple(len(inputs))
        graphs_list = graphs_combined_with_signals(basal_graph_with_depth, input_signals)

        for depth_level, indices in enumerate(depth_groups):
            graphs_list = update_train_mask(graphs_list, indices)

            # 预测当前级别的wire节点输出
            for graph in graphs_list:
                predictions = predict(model, graph, device)
                # 更新节点特征
                for wire_index in depth_groups[depth_level]:
                    if predictions[wire_index]:
                        new_feature = [0, 0, 0, 1, 0, 0, 0, 1, 0]
                    else:
                        new_feature = [0, 0, 0, 1, 0, 0, 1, 0, 0]
                    print(f'In depth{depth_level}, output signal of node{wire_index} is {predictions[wire_index]}')

                    update_node_features(graph, wire_index, new_feature)
