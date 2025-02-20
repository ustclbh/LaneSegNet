import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
from ..builder import NECKS

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
                # 根据公式计算注意力权重
        self.fc = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        
    def forward(self, x, adj, mode):
        
        if mode == "global":
            # x: [N, L, C]，N是batch size，L是节点数，C是通道数
            # adj: [N, L, L]，邻接矩阵
            adj = self.generate_normalized_weight_matrix(x, adj)
            support = torch.matmul(adj, x)  # [N, L, C]
            support = self.linear(support)
        elif mode == "local":
            #x: [N, K, L, C]
            #adj: [N, K, L, L]
            outputs = []
            for i in range(x.shape[1]):
                adj_i = adj[:, i, :, :]
                x_i = x[:, i, :, :]
                adj_i = self.generate_normalized_weight_matrix(x_i, adj_i)
                support = torch.matmul(adj_i, x_i)  # [N, L, C]
                support = self.linear(support)
                outputs.append(support)
            support = torch.stack(outputs, dim=1)
        return support
    
    def generate_normalized_weight_matrix(self, x, adj):
        # 假设节点特征为全1张量，实际应用中应从特征图中提取
        num_nodes = adj.size(1)
        node_features = x
        attention_weights = self.calculate_attention_weights(node_features, adj)
        normalized_weight_matrix = self.normalize_attention_weights(attention_weights)
        return normalized_weight_matrix


    def calculate_attention_weights(self, node_features, adj):
            batch_size, num_nodes, _ = node_features.size()
            attention_weights = []
            for i in range(batch_size):
                weight_matrix = []
                for j in range(num_nodes):
                    row = []
                    for k in range(num_nodes):
                        if adj[i, j, k] == 1:
                            concat_features = torch.cat((node_features[i, j], node_features[i, k]), dim=0)
                            a_jk = self.fc(concat_features)
                            row.append(a_jk)
                        else:
                            row.append(0)
                    weight_matrix.append(row)
                weight_matrix = torch.tensor(weight_matrix).to(adj.device)
                attention_weights.append(weight_matrix)
            attention_weights = torch.stack(attention_weights)
            return attention_weights
    # 归一化注意力权重
    def normalize_attention_weights(self, attention_weights):
        normalized_weights = []
        for i in range(attention_weights.size(0)):
            row_sums = attention_weights[i].sum(dim=1, keepdim=True)
            norm_row = attention_weights[i] / row_sums
            normalized_weights.append(norm_row)
        normalized_weights = torch.stack(normalized_weights)
        return normalized_weights




class GraphEnhancementBlock(nn.Module):
    def __init__(self, in_channels, mode="global"):
        super(GraphEnhancementBlock, self).__init__()
        self.mode = mode
        self.graph_conv1 = GraphConv(in_channels, in_channels)
        self.ln1 = nn.LayerNorm(in_channels)
        self.relu1 = nn.ReLU()
        self.graph_conv2 = GraphConv(in_channels, in_channels)
        self.ln2 = nn.LayerNorm(in_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x, adj):
        # x: [N, L, C]，N是batch size，L是节点数，C是通道数
        # adj: [N, L, L]，邻接矩阵
        identity = x;
        out = self.graph_conv1(x, adj, self.mode);
        out = self.ln1(out)
        out = self.relu1(out)
        out = self.graph_conv2(out, adj, self.mode)
        out = self.ln2(out)
        out = self.relu2(out)
        out += identity
        return out

@NECKS.register_module()
class GraphEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(GraphEnhancementModule, self).__init__()
        self.local_graph_enhancement_block = GraphEnhancementBlock(in_channels, mode="local")
        self.global_graph_enhancement_block = GraphEnhancementBlock(in_channels, mode="global")
    def forward(self, ref_nodes, flat_lane_nodes, gbl_adj_matrix, loc_adj_matrix):
        init_ref_nodes = ref_nodes
        init_flat_lane_nodes = flat_lane_nodes
        ref_nodes = self.global_graph_enhancement_block(ref_nodes, gbl_adj_matrix)
        flat_lane_nodes = self.local_graph_enhancement_block(flat_lane_nodes, loc_adj_matrix)
        ref_nodes = ref_nodes + init_ref_nodes
        flat_lane_nodes = flat_lane_nodes + init_flat_lane_nodes
        return ref_nodes, flat_lane_nodes