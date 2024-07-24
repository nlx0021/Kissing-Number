import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))


class Attention(nn.Module):
    
    def __init__(self,
                 x_channel=3,      # dim
                 y_channel=3,      # dim
                 N_x=15,           # N_limit
                 N_y=15,           # N_limit
                 causal_mask=False,
                 N_heads=16,
                 d=32,
                 w=4,
                 device='cuda'):
        
        super(Attention, self).__init__()
        self.x_channel, self.y_channel = x_channel, y_channel
        self.N_x, self.N_y = N_x, N_y
        self.causal_mask = causal_mask
        self.N_heads = N_heads
        self.d, self.w = d, w
        self.device = device
        
        self.x_layer_norm = nn.LayerNorm(x_channel)
        self.y_layer_norm = nn.LayerNorm(y_channel)
        self.final_layer_norm = nn.LayerNorm(x_channel)
        self.W_Q = nn.Linear(x_channel, d * N_heads, bias=False)
        self.W_K = nn.Linear(y_channel, d * N_heads, bias=False)
        self.W_V = nn.Linear(y_channel, d * N_heads, bias=False)
        self.linear_1 = nn.Linear(d * N_heads, x_channel)
        self.linear_2 = nn.Linear(x_channel, x_channel * w)
        self.linear_3 = nn.Linear(x_channel * w, x_channel)
        self.gelu = nn.GELU()
        
        
    def forward(self, x):
        
        # Input:
        #   [x, (y)]. If y is missed, y=x. 
        
        N_heads = self.N_heads
        N_x, N_y = self.N_x, self.N_y
        d, w = self.d, self.w
        
        if len(x) == 1:
            x = x[0]
            y = x.clone()
        else:
            x, y = x
        
        batch_size = x.shape[0]
        
        x_norm = self.x_layer_norm(x)     # [B, N_x, c_x]
        y_norm = self.y_layer_norm(y)     # [B, N_y, c_y]
        
        q_s = self.W_Q(x_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_x, d]
        k_s = self.W_K(y_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_y, d]
        v_s = self.W_V(y_norm).view(batch_size, -1, N_heads, d).transpose(1, 2)   # [batch_size, N_heads, N_y, d]
        
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(d)            # [batch_size, N_heads, N_x, N_y]
        # scores = F.softmax(scores, dim=-1)
        if self.causal_mask:
            mask = (torch.from_numpy(np.triu(np.ones([batch_size, N_heads, N_x, N_y]), k=1)) == 0).to(self.device)
            scores = scores.masked_fill(mask == 0, -1e9)
            # scores = scores * torch.from_numpy(np.triu(np.ones([batch_size, N_heads, N_x, N_y]), k=1)).float().to(self.device)
        scores = F.softmax(scores, dim=-1)
        
        o_s = torch.matmul(scores, v_s)   # [batch_size, N_heads, N_x, d]
        o_s = o_s.transpose(1, 2).contiguous().view(batch_size, -1, N_heads*d)          # [batch_size, N_x, N_heads*d]
        x = x.reshape(batch_size*N_x, -1) + self.linear_1(o_s.reshape(-1, N_heads*d))   # [batch_size*N_x, c_x]
        
        x = (x + self.linear_3(self.gelu(self.linear_2(self.final_layer_norm(x))))).reshape(batch_size, N_x, -1)     # [batch_size, N_x, c_x]
        
        return x


class PolicyHead(nn.Module):
    
    def __init__(self,
                 d=3,
                 N_limit=15,
                 n_attentive=4,
                 N_heads=32,
                 N_features=64,
                 device="cuda"):
        
        self.d = d
        self.N_limit = N_limit
        self.n_attentive = n_attentive
        
        self.attentions = nn.ModuleList(
            [Attention(
                x_channel=d,
                y_channel=d,
                N_x=N_limit,
                N_y=N_limit,
                causal_mask=True,
                N_heads=N_heads,
                d=N_features,
                device=device
            ) for _ in range(n_attentive)]
        )
        
    
    def forward(self, x):
        
        # x: [B, N, d]
        d = self.d
        N_limit = self.N_limit
        n_attentive = self.n_attentive
        
        for idx in range(n_attentive):
            c = self.attentions[idx](x)
            x = x + c
        
        mu = x    # [B, N, d]
        
        return mu
        
        
class ValueHead(nn.Module):
    
    def __init__(self,
                 d=3,
                 N_limit=15,
                 value_layers=3,
                 inter_channel=256,
                 device='cuda'):
        
        self.d = d
        self.N_limit = N_limit
        self.value_layers = value_layers
        self.in_linear = nn.Linear(N_limit*d, inter_channel)
        self.linears = nn.ModuleList(
            [nn.Linear(inter_channel, inter_channel) for _ in range(value_layers-1)]
        )
        self.out_linear = nn.Linear(inter_channel, 2)
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(value_layers)])
        
        
    def forward(self, x):
        
        x = x.reshape(-1, 1)           # [B, N*d]
        
        N_layers = self.value_layers
        x = self.relus[0](self.in_linear(x))
        for idx in range(N_layers-1):
            x = self.relus[idx+1](self.linears[idx](x))
        x = self.out_linear(x)
        
        v, sigma = x[:,:1], x[:,1:]    # [B, 1], [B, 1]
        return v, sigma
        
        

class Net(nn.Module):
    
    def __init__(self,
                 d=3,
                 N_limit=15,
                 n_attentive=4,
                 N_heads=32,
                 N_features=64,
                 value_layers=3,
                 inter_channel=256,
                 device='cuda'):
        
        self.d = d
        self.N_limit = N_limit
        
        self.policy_head = PolicyHead(
            d=d,
            N_limit=N_limit,
            n_attentive=n_attentive,
            N_heads=N_heads,
            N_features=N_features,
            device=device
        )
        
        self.value_head = ValueHead(
            d=d,
            N_limit=N_limit,
            value_layers=value_layers,
            inter_channel=inter_channel,
            device=device
        )
        
    
    def forward(self, x):
        
        mu = self.policy_head(x)
        v, sigma = self.value_head(x)
        
        return mu, v, sigma