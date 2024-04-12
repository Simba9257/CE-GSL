import networkx as nx
import numpy as np
import torch

from utils.functional import community_detection

def reshape(logits, edge_index_2, weight, adj_matrix, k, dataname):
    ##### 求community #####

    # 创建空图
    new_graph = nx.Graph()

    # 添加节点和边
    num_nodes = logits.size(0)
    for j in range(num_nodes):
        new_graph.add_node(j, features=logits[j].tolist())

    num_edges = edge_index_2.t().shape[1]
    for j in range(num_edges):
        src = edge_index_2.t()[0][j].item()
        tgt = edge_index_2.t()[1][j].item()
        weight_ = weight[j].item()
        new_graph.add_edge(src, tgt, weight=weight_)

    communities = community_detection('leiden')(new_graph).communities  # louvain combo leiden ilouvain eigenvector girvan_newman demon lemon lpanni
    # print(communities)

    # 计算每个簇内的度数和
    degree = adj_matrix.sum(dim=1)
    cluster_degrees = {}
    for cluster, node in enumerate(communities):
        for nd in node:
            if cluster not in cluster_degrees:
                cluster_degrees[cluster] = degree[nd]
            else:
                cluster_degrees[cluster] += degree[nd]
    # print('簇内的度数和:',cluster_degrees)

    # 建立节点到簇的索引
    node_to_cluster = {}
    for cluster, nodes in enumerate(communities):
        for node in nodes:
            node_to_cluster[node] = cluster
    # 将稠密张量转换为稀疏张量
    sparse_adj = adj_matrix.nonzero().numpy()
    # 计算簇间的度数和
    inter_cluster_degree = {}
    for u, v in sparse_adj:
        if node_to_cluster[u] != node_to_cluster[v]:
            if node_to_cluster[u] not in inter_cluster_degree:
                inter_cluster_degree[node_to_cluster[u]] = adj_matrix[u][v]
            else:
                inter_cluster_degree[node_to_cluster[u]] += adj_matrix[u][v]
            if node_to_cluster[v] not in inter_cluster_degree:
                inter_cluster_degree[node_to_cluster[v]] = adj_matrix[u][v]
            else:
                inter_cluster_degree[node_to_cluster[v]] += adj_matrix[u][v]
    for cd in cluster_degrees:
        if cd not in inter_cluster_degree:
            inter_cluster_degree[cd] = 0
    # print("各簇内节点与簇外相连边的度数：", inter_cluster_degree)

    VOL = adj_matrix.sum()
    se_dict = {'root': torch.zeros(len(communities))}
    for c, cluster in enumerate(communities):
        #print(inter_cluster_degree, cluster_degrees, se_dict['root'], c)
        se_dict['root'][c] = -(inter_cluster_degree[c] / VOL) * torch.log2((cluster_degrees[c] + 1) / (VOL + 1))
        se = torch.zeros(len(cluster))
        for e, node in enumerate(cluster):
            se[e] = -(degree[node] / VOL) * torch.log2((degree[node] + 1) / (cluster_degrees[c] + 1)) + se_dict['root'][
                c]
        se = torch.softmax(se.float(), dim=0)
        se_dict[c] = se
    se_dict['root'] = torch.softmax(se_dict['root'].float(), dim=0)
    # print(se_dict)

    new_edge_index = []
    for community_id, node_list in enumerate(communities):
        if len(node_list) == 1:
            continue
        prefer_edge_num = round(k * len(node_list))
        # print(prefer_edge_num)
        for edge_num in range(prefer_edge_num):
            se = se_dict[community_id]
            id1, id2 = torch.multinomial(se, num_samples=2, replacement=True)
            link_id1 = node_list[id1]
            link_id2 = node_list[id2]
            link_id = [link_id1, link_id2]
            new_edge_index.append(link_id)
    # 选簇间
    se = se_dict['root']
    prefer_edge_num = round(k * len(se))
    for edge_num in range(prefer_edge_num):
        cl1, cl2 = torch.multinomial(se, num_samples=2, replacement=True)
        id1 = torch.multinomial(se_dict[cl1.item()], num_samples=1, replacement=False)
        id2 = torch.multinomial(se_dict[cl2.item()], num_samples=1, replacement=False)
        link_id1 = communities[cl1][id1]
        link_id2 = communities[cl2][id2]
        link_id = [link_id1, link_id2]
        new_edge_index.append(link_id)
    new_edge_index = torch.tensor(new_edge_index)
    new_edge_index = torch.concat((new_edge_index, torch.flip(new_edge_index, dims=[1])), dim=0)
    new_edge_index = torch.unique(new_edge_index, dim=0).t()
    # print(new_edge_index)

    return new_edge_index

