import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SelectiveAttentionModule(nn.Module):
  def __init__(self , embd_dim , k_dim , v_dim, seq_len , mask = False):
    super(SelectiveAttentionModule , self).__init__()
    self.mask = mask
    self.seq_len = seq_len
    self.Wq = nn.Linear(embd_dim , k_dim)
    self.Wk = nn.Linear(embd_dim , k_dim)
    self.Wv = nn.Linear(embd_dim , v_dim)
    self.alpha = nn.Parameter(torch.randn(1 , seq_len))
    self.tokenq = nn.Parameter(torch.randn(1 , seq_len , k_dim))
    self.tokenv = nn.Parameter(torch.randn(1 , seq_len , v_dim))

  def forward(self , embedding_matrix): #Embedding matrix is a 3D tensor of shape (batch_size , seq_len , embd_dim)
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.expand(batch_size , -1 , -1)
    tokenv = self.tokenv.expand(batch_size , -1 , -1)
    alpha = self.alpha.expand(batch_size , -1 , -1)

    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(torch.diagonal(pdt_q , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)
    token_temp_v = F.tanh(torch.diagonal(pdt_v , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device)

    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices)

    temps_q = position_temp + token_temp_q
    temps_v = position_temp + token_temp_v

    gated_Q = temps_q.unsqueeze(-1) * Q
    gated_V = temps_v.unsqueeze(-1) * V

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return torch.bmm(attention_scores , gated_V)

  def attention_map(self , embedding_matrix):
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.expand(batch_size , -1 , -1)
    tokenv = self.tokenv.expand(batch_size , -1 , -1)
    alpha = self.alpha.expand(batch_size , -1 , -1)

    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(torch.diagonal(pdt_q , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)
    token_temp_v = F.tanh(torch.diagonal(pdt_v , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device)

    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices)

    temps_q = position_temp + token_temp_q
    temps_v = position_temp + token_temp_v

    gated_Q = temps_q.unsqueeze(-1) * Q
    gated_V = temps_v.unsqueeze(-1) * V

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return attention_scores

  def token_aware_temperatures(self , embedding_matrix):
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.expand(batch_size , -1 , -1)
    tokenv = self.tokenv.expand(batch_size , -1 , -1)
    alpha = self.alpha.expand(batch_size , -1 , -1)

    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(torch.diagonal(pdt_q , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)
    token_temp_v = F.tanh(torch.diagonal(pdt_v , dim1 = 1 , dim2 = 2)) #(batch_size , seq_len)

    return token_temp_q ,token_temp_v

  def position_aware_temperatures(self , embedding_matrix):
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.expand(batch_size , -1 , -1)
    tokenv = self.tokenv.expand(batch_size , -1 , -1)
    alpha = self.alpha.expand(batch_size , -1 , -1)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device)

    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices) #(batch_size , seq_len)
    return position_temp