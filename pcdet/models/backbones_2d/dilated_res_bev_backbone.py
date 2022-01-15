from operator import mod
import numpy as np
import torch
import torch.nn as nn

class dilated_block(nn.Module):
    '''dilated block for bev feature extraction
    Args:
        in_channels: input channles
        mid_channels: 
        dilation: 

    '''
    def __init__(self, in_channels, mid_channels, dilation):
        super(dilated_block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=mid_channels,
                kernel_size=1,
                dilation=dilation,
            ),
            nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=in_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(in_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )

    def forward(self, x):
        idensity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + idensity
        return out

class DilatedResBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(DilatedResBEVBackbone, self).__init__()
    
        self.in_channels = input_channels
        self.out_channels = model_cfg.OUTPUT_CHANNELS
        self.block_mid_channels = model_cfg.MID_CHANNELS
        self.num_res_blocks = model_cfg.NUM_RES_BLOCKS
        self.block_dilations = model_cfg.BLOCK_DILATIONS
        self.num_bev_features = self.out_channels

        assert self.num_res_blocks == len(self.block_dilations), "the number of residual blocks should be equal to the length of block dilations"
        
        # init layers
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.block_mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.block_mid_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    
        cur_layer = []
        for i in range(self.num_res_blocks):
            cur_layer.append(
                dilated_block(in_channels=self.block_mid_channels, 
                            mid_channels=self.block_mid_channels,
                            dilation=self.block_dilations[i])
            )

        self.mid_layer = nn.Sequential(*cur_layer)
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.block_mid_channels, out_channels=self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
    
    def forward(self, data_dict):
        x = data_dict['spatial_features']
        out = self.input_layer(x)
        out = self.mid_layer(out)
        out = self.out_layer(out)
        data_dict['spatial_features_2d'] = out
        return data_dict



if __name__ == "__main__":
    # layer = DilatedBevBackbone(256, 512, 512, 4, [2, 4, 6, 8])
    # a = torch.randn(1, 256, 200, 176)
    # print(a.shape)
    # out = layer(a)
    # print(out.shape)
    pass