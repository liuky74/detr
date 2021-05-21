
import torch
from torch.nn import functional as F
from torch import nn
import math



def attention(query, key, value, mask=None, dropout=None):
    " 将src+pos的数据自乘,实际得到的就是src中每个像素信息之间的联系"
    head_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim) # 将key转置后与query矩阵乘
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)  # 矩阵乘后激活
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # 将元素相关性矩阵与原始数据矩阵乘


# Q,K,V shape: [batch_size,28*38,256]
query = key = value = torch.randn(2,28*38,256)


linear_q=nn.Linear(256,256)
linear_k=nn.Linear(256,256)
linear_v=nn.Linear(256,256)
final_linear = nn.Linear(256,256)
query = linear_q(query)
query = query.view(2,-1,8,32)
query = torch.transpose(query,1,2)

key = linear_k(key)
key = key.view(2,-1,8,32)
key = torch.transpose(key,1,2)  # 形成shape:[batch_size,head_num,h*w,head_dim]的数据
value = linear_v(value)
value = value.view(2,-1,8,32)
value = torch.transpose(value,1,2)

value,src_att = attention(query,key,value) # 返回了添加相关性的src,以及相关性矩阵

value = torch.transpose(value,1,2)  # 重新形成[batch,h*w,head_num,head_dim]的数据

res = final_linear(value)



print("query")
