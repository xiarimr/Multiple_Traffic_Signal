# 复现论文RDP-LSTM使用的工具
import os
import json
import numpy as np

def get_adj_mat(path_to_work_directory, dic_traffic_env_conf):
    num_col = dic_traffic_env_conf["NUM_COL"]
    num_row = dic_traffic_env_conf["NUM_ROW"]
    num_intersections = num_col * num_row
    # 初始化为对角矩阵
    vector = np.array([1] * num_intersections, dtype=int)
    inter_adj_mat = np.diag((vector))

    file = os.path.join(path_to_work_directory, dic_traffic_env_conf["ROADNET_FILE"])
    with open("{0}".format(file)) as json_data:
        net = json.load(json_data)
        intersections = []
        for inter in net["intersections"]:
            if not inter["virtual"]:
                intersections.append(inter["id"])
        # 建立 ID 到索引映射
        id_to_index = {inter_id: i for i, inter_id in enumerate(intersections)}
        for roads in net["roads"]:
            start_id = roads["startIntersection"]
            end_id = roads["endIntersection"]
            if start_id in id_to_index and end_id in id_to_index:
                i = id_to_index[start_id]
                j = id_to_index[end_id]
                inter_adj_mat[i][j] = 1
    return inter_adj_mat, intersections
