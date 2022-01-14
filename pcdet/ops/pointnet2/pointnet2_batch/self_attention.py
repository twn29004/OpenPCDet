import numpy as np
from numpy.lib.function_base import select
import torch
from torch import nn
from torch.nn import init
from pointnet2_utils import ball_query

def generate_position_mask(radius: float, nsample: int, rois_position: torch.Tensor):
    '''
    :param radius: search radius
    :param nsample: max sample number
    :param rois_position: the position of rois, (batch_size, num_rois, num_channels) 
    :return: mask (num_rois, num_rois)
    '''
    idx = ball_query(radius, nsample, rois_position, rois_position).long()
    mask = rois_position.new_zeros(rois_position.shape[0], rois_position.shape[1], rois_position.shape[1])
    value = rois_position.new_ones(rois_position.shape[0], rois_position.shape[1], rois_position.shape[1])
    mask.scatter_(dim=1, index=idx, src=value)
    return mask


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, 0) # 将mask部分填充为0
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class OffsetAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=.1):
        '''this module doesn't support multi-head
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        '''
        super(OffsetAttention, self).__init__()
        self.fc_q = nn.Linear(d_model,  d_k)
        self.fc_k = nn.Linear(d_model, d_k)
        self.fc_v = nn.Linear(d_model, d_v)
        self.fc_o = nn.Linear(d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.activate = nn.ReLU()
        self.after_norm = nn.BatchNorm1d(d_model)
        self.fc_out = nn.Linear(d_model, d_model)


        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        ''' 
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        q = self.fc_q(queries) #(bs, n_q, d_k)
        k = self.fc_k(keys).transpose(1,2) #(bs, d_k, n_k)
        v = self.fc_v(values) # (bs, n_k, d_v)

        att = torch.matmul(q, k) # (bs, n_q, n_k)
    
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, 0) # 将mask部分填充为0

        att = torch.softmax(att, -1)
        att = self.dropout(att)
        att = att / (1e-9 + att.sum(dim=1, keepdims=True))

        out = torch.matmul(att, v)
        print((values - out).shape)
        out = self.fc_out(values - out).transpose(1,2)
        out = self.after_norm(out).transpose(1,2)
        out = self.activate(out)
        return out



if __name__ == "__main__":
    # rois = torch.randn(2, 128, 512).cuda()
    # rois_position = torch.randn(2, 128, 16) * 20
    # rois_position = rois_position.cuda()
    # mask = generate_position_mask(radius=8, nsample=3, rois_position=rois_position).unsqueeze(1)
    # mask = (mask == 0)
    # selfAttention = ScaledDotProductAttention(d_model=512, d_k=(512  // 8), d_v=512, h=16).cuda()
    # output = rois = selfAttention(rois, rois, rois, attention_mask=mask)
    # print(output.shape)
    rois = torch.randn(2, 128, 512).cuda()
    offset_attention = OffsetAttention(512, int(512 // 4), 512).cuda()
    output = offset_attention(rois, rois, rois)
    print(output.shape)
    
    
    
    