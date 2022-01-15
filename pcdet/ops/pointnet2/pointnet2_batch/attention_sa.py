from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, input_channels=3, output_channels=64):
        super().__init__()
        self.embeding = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )
    def forward(self, xyz):
        """
        Args:
            xyz: B, N, 3
        Returns:
            input_embedding: B, C, N
        """
        xyz = xyz.permute(0, 2, 1) #(B, npoints, c) -> (B, c, npoints)
        new_xyz = self.embeding(xyz)
        new_xyz = new_xyz #(B, output_channels, npoint)
        return new_xyz


class OffsetAttention(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels: input channels 
        Returns:

        """
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False).cuda()
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False).cuda()
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1).cuda()
        self.trans_conv = nn.Conv1d(channels, channels, 1).cuda()
        self.after_norm = nn.BatchNorm1d(channels).cuda()
        self.activate = nn.ReLU().cuda()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input_embedding):
        """
        Args:
            input_embedding: B, C, npoints
        Returns:
            new_features: B, C, npoints
        """
        # (B, npoints, C) -> (B, C, npoints)
        Q = self.q_conv(input_embedding).permute(0, 2, 1) # (B, npoints, C // 4)
        K = self.k_conv(input_embedding) # (B, C // 4, npoints)
        V = self.v_conv(input_embedding) # (B, C, npoints)
        attention_map = self.softmax(Q @ K) # (B, N, N)
        attention_map = attention_map / (1e-9 + attention_map.sum(dim=1, keepdims=True))
        new_features = V @ attention_map # (B, C, N)
        new_features = self.activate(self.after_norm(self.trans_conv(input_embedding - new_features))) # (B, C, N)
        return new_features
    
class StackedPointAttention(nn.Module):
    def __init__(self, channels_in = None, channels_out = None, attention_method = None, num_attention_layer = None):
        """
        Args:
            channels: point features channels
            attention_method: SelfAttenion or OffsetAttenion
            num_attenion_layer: 
        Returns: 
        """
        if attention_method == "SelfAttention":
            attention_method = SelfAttenion
        elif attention_method == "OffsetAttention":
            attention_method = OffsetAttention
        else:
            raise NotImplementedError
        super().__init__()
        self.num_attenion_layer = num_attention_layer
        self.LBR = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.ReLU(),
            nn.Conv1d(channels_out, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels_out),
            nn.ReLU()
        )
        self.attention_layer_list = []
        for i in range(self.num_attenion_layer):
            self.attention_layer_list.append(attention_method(channels_out))
    
    def forward(self, features):
        """
        Args:
            features: B, C, N
        Returns:
            new_features: B, C * num_attention_layer, N 
        """
        features = self.LBR(features)
        new_features_list = [features]
        for attention in self.attention_layer_list:
            new_features_list.append(attention(new_features_list[-1]))
    
        new_feature = torch.cat(new_features_list[1:], dim = 1) # (B, nnum_attention_layer * channels, npoints)
        return new_feature

class PCTEncoder(nn.Module):
    def __init__(self, input_xyz_channel, input_embeding_channels, features_channels, use_features=False, \
                 attention_method : str = "SelfAttention", attention_layer_num : int = 4,\
                 pool_method : str = "max_pool", output_channels : int = None):
        """
        Args:
            input_xyz_channels: 输入的xyz的维度
            input_embeding_channels: 输入嵌入所需的维度
            features_channels: 逐点的特征的维度
            use_features: 是否使用features
            attention_method: SelfAttenion和OffsetAttention的区别
            attention_layer_num: attention的层数 
        """
        super().__init__()
        self.use_features = use_features
        self.pool_method = pool_method

        self.input_embedding = InputEmbedding(input_channels=input_xyz_channel, output_channels=input_embeding_channels)
        if use_features:
            features_channels += input_embeding_channels

        self.stack_attention = StackedPointAttention(features_channels, attention_method, attention_layer_num)
    
        self.output_layer = nn.Sequential(
            nn.Conv1d(features_channels * attention_layer_num, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU()
        )

    def forward(self, xyz, features):
        """
        Args:
            xyz: B, N, 3
            features: B, C, N  
        """
        input_embedding = self.input_embedding(xyz) # (B, input_embedding_channels, N)
        if features is not None:
            assert self.use_features == True, "features is not None, use_features should be True"
            input_embedding = torch.cat([input_embedding, features], dim=1)
        else:
            assert self.use_features == False, "feature is None, use_features should be False"
        new_features = self.stack_attention(input_embedding) # (B, C, N)

        if self.pool_method == "max_pooling":
            new_features = torch.max(new_features, dim=-1)[0] # (B, C)
        elif self.pool_method == "avg_pooling":
            new_features = torch.mean(new_features, dim=-1)
        else:
            raise NotImplementedError
        
        new_features = self.output_layer(new_features.unsqueeze(dim=-1))
        return new_features


if __name__ == "__main__":
    Sa = PCTEncoder(3, 64, 128, True, "OffsetAttention", 4, "max_pooling", 512)
    Sa.cuda()
    points = torch.randn(128, 512, 3).cuda()
    features = torch.randn(128, 128, 512).cuda()
    new_features = Sa(points, features)
    print(new_features.shape)

