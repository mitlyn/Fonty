import torch, torch.nn as nn

from model.blocks import *

# *----------------------------------------------------------------------------*


class SelfAttention(nn.Module):
    def __init__(self, dim: int):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels = dim, out_channels = dim // 8, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = dim, out_channels = dim // 8, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = dim, out_channels = dim, kernel_size = 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, X):
        """
            inputs :
                x : input feature maps(B × C × W × H)
            returns :
                out : self attention value + input feature
                attention: B × N × N (N is Width*Height)
        """
        m_batchsize, C, width, height = X.size()
        proj_query = self.query_conv(X).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(X).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(X).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + X

        return out
