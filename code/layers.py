import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_size, out_size, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_size
        self.out_features = out_size
        '''参数定义'''
        self.weight = Parameter(torch.FloatTensor(in_size, out_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))
        else:
            self.register_parameter('bias', None)
        '''初始化参数'''
        self.reset_parameters()

    """生成权重"""
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    """
    前馈运算 即计算A~ X W(0)
    input X与权重W相乘，然后adj矩阵与他们的积稀疏乘
    直接输入与权重之间进行torch.mm操作，得到support，即XW
    support与adj进行torch.spmm操作，得到output，即AXW选择是否加bias
    """
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 矩阵乘法
        output = torch.spmm(adj, support)  # 稀疏矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    # 通过设置断点，可以看出output的形式是0.01，0.01，0.01，0.01，0.01，#0.01，0.94]，里面的值代表该x对应标签不同的概率
    # 故此值可转换为#[0,0,0,0,0,0,1]，对应我们之前把标签onehot后的第七种标签

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Highway_dense(Module):
    """
    y = t * h1 + (1 - t) * h2
    h2 = h2*t+(1-t)*h1
    """
    def __init__(self, in_size, out_size):
        super(Highway_dense, self).__init__()

        self.in_features = in_size
        self.out_features = out_size
        self.gconv = GraphConvolution(in_size, out_size)
        self.linear = nn.Linear(in_size, out_size, bias=True)
        '''初始化参数'''
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant(self.linear.bias, -4.0)

    def forward(self, x, adj):
        # graphconv layer
        l_h = self.gconv(x, adj)
        l_h = F.tanh(l_h)

        # gate layer
        l_t = self.linear(x)
        l_t = F.sigmoid(l_t)

        #   y = t * h1 + (1 - t) * h2
        output = l_t * l_h + (1.0 - l_t) * x

        return output
