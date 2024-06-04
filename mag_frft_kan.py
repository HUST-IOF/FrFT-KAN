import math
from torch_frft.frft_module import frft
import torch
import torch.nn as nn
import torch.nn.functional as F



class FRFTLinear(nn.Module):
    def __init__(self, in_features, out_features, order=1.0):
        super(FRFTLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.order = order

        # Linear weights for combining outputs
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def frft_transform(self, x):
        transformed_x = frft(x, self.order, dim=-1)
        return transformed_x

    def forward(self, x):
        transformed_x = torch.abs(self.frft_transform(x))
        output = F.linear(transformed_x, self.weight)
        return self.bn(output)


class FRFTNet(nn.Module):
    def __init__(self, layers_hidden, alpha):
        super(FRFTNet, self).__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(FRFTLinear(in_features, out_features, alpha))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

