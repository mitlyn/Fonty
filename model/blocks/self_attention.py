import torch, torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // 8, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B × C × W × H)
            returns :
                out : self attention value + input feature
                attention: B × N × N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B × C × (N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)                       # B × C × (W * H)
        energy = torch.bmm(proj_query, proj_key)                                                # transpose check
        attention = self.softmax(energy)                                                        # B × (N) × (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)                   # B × C × N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x

        return out
