import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class SelectiveAttentionModule(nn.Module):
  def __init__(self , embd_dim , k_dim , v_dim , mask = False):
    super(SelectiveAttentionModule , self).__init__()
    self.mask = mask
    self.Wq = nn.Linear(embd_dim , k_dim)
    self.Wk = nn.Linear(embd_dim , k_dim)
    self.Wv = nn.Linear(embd_dim , v_dim)
    self.alpha = nn.Parameter(torch.empty(1 ,))
    self.tokenq = nn.Parameter(torch.empty(1 , k_dim))
    self.tokenv = nn.Parameter(torch.empty(1 , v_dim))

    self._init_parameters()

  def _init_parameters(self):
    nn.init.xavier_uniform_(self.tokenq)
    nn.init.xavier_uniform_(self.tokenv)
    nn.init.constant_(self.alpha, 0.0)

  def forward(self , embedding_matrix): 
    """
    Makes a forward pass through the attention module, given a 3D embedding matrix of shape 
    (batch_size , seq_len , embd_dim)

    Returns tensor of context aware value vectors of shape (batch_size , seq_len , v_dim)
    """
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.view(1,1,dk).repeat(batch_size,1,1) #(batch_size , 1 , k_dim)
    tokenv = self.tokenv.view(1,1,dv).repeat(batch_size,1,1) #(batch_size , 1 , v_dim)
    alpha = self.alpha.view(1,1,1).repeat(batch_size,seq_len,1) # (batch_size , seq_len , 1)

    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(pdt_q.squeeze(1)) #(batch_size , seq_len)
    token_temp_v = F.tanh(pdt_v.squeeze(1)) #(batch_size , seq_len)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device).view(1,seq_len,1)

    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices) #(batch_size , seq_len , 1)
    position_temp = position_temp.squeeze(-1) #(batch_size , seq_len)

    temps_q = position_temp + token_temp_q #(batch_size , seq_len)
    temps_v = position_temp + token_temp_v #(batch_size , seq_len)

    gated_Q = temps_q.unsqueeze(-1).repeat(1,1,dk) * Q #(batch_size , seq_len , kdim)
    gated_V = temps_v.unsqueeze(-1).repeat(1,1,dv) * V #(batch_size , seq_len , vdim)

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return torch.bmm(attention_scores , gated_V)

  def attention_map(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns a tensor of attention maps, 
    of shape (batch_size , seq_len , seq_len) 
    """
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.view(1,1,dk).repeat(batch_size,1,1) #(batch_size , 1 , k_dim)
    tokenv = self.tokenv.view(1,1,dv).repeat(batch_size,1,1) #(batch_size , 1 , v_dim)
    alpha = self.alpha.view(1,1,1).repeat(batch_size,seq_len,1) # (batch_size , seq_len , 1)

    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(pdt_q.squeeze(1)) #(batch_size , seq_len)
    token_temp_v = F.tanh(pdt_v.squeeze(1)) #(batch_size , seq_len)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device).view(1,seq_len,1)

    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices) #(batch_size , seq_len , 1)
    position_temp = position_temp.squeeze(-1) #(batch_size , seq_len)

    temps_q = position_temp + token_temp_q #(batch_size , seq_len)
    temps_v = position_temp + token_temp_v #(batch_size , seq_len)

    gated_Q = temps_q.unsqueeze(-1).repeat(1,1,dk) * Q #(batch_size , seq_len , kdim)
    gated_V = temps_v.unsqueeze(-1).repeat(1,1,dv) * V #(batch_size , seq_len , vdim)

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)
    return attention_scores

  def token_aware_temperatures(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns two tensors of shape 
    (batch_size , seq_len) of token aware temperatures correponding to query and value vectors
    """
    
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = K.shape[1]

    tokenq = self.tokenq.view(1,1,dk).repeat(batch_size,1,1) #(batch_size , 1 , k_dim)
    tokenv = self.tokenv.view(1,1,dv).repeat(batch_size,1,1) #(batch_size , 1 , v_dim)
    
    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2)))
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2)))

    token_temp_q = F.tanh(pdt_q.squeeze(1)) #(batch_size , seq_len)
    token_temp_v = F.tanh(pdt_v.squeeze(1)) #(batch_size , seq_len)

    return token_temp_q ,token_temp_v

  def position_aware_temperatures(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns a tensor of shape 
    (batch_size , seq_len) of position aware temperatures correponding to query and value vectors
    """
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = K.shape[1]

    alpha = self.alpha.view(1,1,1).repeat(batch_size,seq_len,1) # (batch_size , seq_len , 1)

    indices = torch.arange(1 , seq_len+1 , device = alpha.device).view(1,seq_len,1)
    
    position_temp = 1 + torch.sigmoid(alpha) * torch.log(indices) #(batch_size , seq_len , 1)
    position_temp = position_temp.squeeze(-1) #(batch_size , seq_len)
    
    return position_temp
