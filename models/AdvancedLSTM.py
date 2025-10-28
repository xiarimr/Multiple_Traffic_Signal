import tensorflow as tf
from tensorflow import matmul, sigmoid, transpose
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np



class MLP(layers.Layer):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc1 = layers.Dense(feature_dim, kernel_initializer='random_normal', activation='relu')
        self.fc2 = layers.Dense(feature_dim, kernel_initializer='random_normal', activation='relu')
    
    def call(self, obs):
        e = self.fc1(obs)
        e = self.fc2(e)
        return e


class NeighborAttention(layers.Layer):
    def __init__(self, feature_dim):
        super().__init__()
        self.Wq = layers.Dense(feature_dim, use_bias=False)
        self.Wk = layers.Dense(feature_dim, use_bias=False)
    

    def call(self, x, adj_matrix):
        """
        x: [N, feature_dim]
        adj_matrix: [N, N] (0/1)
        返回 G: [N, N]，按行归一化（行和 = 1，在邻接范围内）
        """

        Q = self.Wq(x)
        K = self.Wk(x)

        alpha_ = matmul(Q, transpose(K))
        adj_float = tf.cast(adj_matrix, tf.float32)
        alpha_ = alpha_ * adj_float
        
        row_sums = tf.reduce_sum(alpha_, axis=1, keepdims=True)
        row_sums = tf.maximum(row_sums, 1e-8)
        
        G = alpha_ / row_sums * adj_float
        
        return G
        

class RDP_LSTMCell_MultiHead(layers.Layer):
    def __init__(self, feature_dim, hidden_dim, num_heads, num_inter):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_inter = num_inter

        self.W_ig = layers.Dense(hidden_dim, use_bias=True)
        self.W_fg = layers.Dense(hidden_dim, use_bias=True)
        self.W_og = layers.Dense(hidden_dim, use_bias=True)
        self.W_C = layers.Dense(hidden_dim, use_bias=True)

        self.W_v = [layers.Dense(hidden_dim, use_bias=False) for _ in range(num_heads)]
        self.attn_heads = [NeighborAttention(hidden_dim + feature_dim) for _ in range(num_heads)]

        self.fc_aig_heads = [layers.Dense(hidden_dim, use_bias=True, activation='relu') for _ in range(num_heads)]
        self.W_aig_heads = [layers.Dense(1, use_bias=False) for _ in range(num_heads)]

    def call(self, e, h_prev, C_prev, G_prev, adj_mask):
        concat_input = tf.concat([h_prev, e], axis=-1)

        inputGates = tf.sigmoid(self.W_ig(concat_input))
        forgetGates = tf.sigmoid(self.W_fg(concat_input))
        outputGates = tf.sigmoid(self.W_og(concat_input))
        cellCandidate = tf.tanh(self.W_C(concat_input))

        C_current = forgetGates * C_prev + inputGates * cellCandidate
        _h = outputGates * tf.tanh(C_current)

        G_heads_updated = []
     
        for h_idx in range(self.num_heads):

            G_tilde = self.attn_heads[h_idx](concat_input, adj_mask)
            
            # 每个头计算自己的门控
            aig_hidden = self.fc_aig_heads[h_idx](concat_input)
            aig = tf.sigmoid(self.W_aig_heads[h_idx](aig_hidden)) 
            afg = 1.0 - aig  # [N, 1]
            
            # 每个头独立更新邻接矩阵
            G_head_updated = afg * G_prev + aig * G_tilde 
            G_heads_updated.append(G_head_updated)
        
        # 合并所有头的邻接矩阵（平均）
        G_current = tf.add_n(G_heads_updated) / self.num_heads

        head_outputs = []
        for h_idx in range(self.num_heads):
            # 每个头用自己更新后的 G 聚合邻居信息
            aggregated = tf.matmul(G_heads_updated[h_idx], _h)
            head_outputs.append(self.W_v[h_idx](aggregated))
        
        # 平均所有头的输出
        h_current = tf.sigmoid(tf.add_n(head_outputs) / self.num_heads)

        return h_current, C_current, G_current


class RDP_LSTM_Model(Model):
    def __init__(self, dic_RDP_LSTM_Model=None):
        super().__init__()
        self.feature_dim = dic_RDP_LSTM_Model['feature_dim']
        self.hidden_dim = dic_RDP_LSTM_Model['hidden_dim']
        self.num_intersection = dic_RDP_LSTM_Model['num_intersection']
        self.action_dim = dic_RDP_LSTM_Model["action_dim"]
        self.num_heads = dic_RDP_LSTM_Model["num_heads"]
        
        self.encoder = MLP(self.feature_dim)
        self.cell = RDP_LSTMCell_MultiHead(self.feature_dim, self.hidden_dim, self.num_heads, self.num_intersection)
        self.output_layer = layers.Dense(self.action_dim, use_bias=False)
    
    def call(self, obs_seq, adj_mask):
        """
        obs_seq: [T, N, obs_dim]
        adj_mask: [N, N]
        """
        N = self.num_intersection

        # 初始化（添加 dtype）
        h = tf.zeros((N, self.hidden_dim), dtype=tf.float32)
        C = tf.zeros((N, self.hidden_dim), dtype=tf.float32)
        G = tf.eye(N, dtype=tf.float32)

        # 修正：使用 tf.unstack 避免动态形状问题
        seq_list = tf.unstack(obs_seq, axis=0)  # list of [N, obs_dim]
        
        for obs_t in seq_list:
            e = self.encoder(obs_t)  # [N, feature_dim]
            h, C, G = self.cell(e, h, C, G, adj_mask)

        # 输出 Q 值
        q_values = self.output_layer(h)  # [N, action_dim]
        return q_values

        
