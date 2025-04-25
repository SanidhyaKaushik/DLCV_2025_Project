import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ModifiedSelectiveAttentionModule(nn.Module):
  def __init__(self , embd_dim , k_dim , v_dim, seq_len , mask = False):
    super(ModifiedSelectiveAttentionModule , self).__init__()
    self.mask = mask
    self.seq_len = seq_len
    self.Wq = nn.Linear(embd_dim , k_dim)
    self.Wk = nn.Linear(embd_dim , k_dim)
    self.Wv = nn.Linear(embd_dim , v_dim)
    self.alphaq = nn.Parameter(torch.empty(seq_len ,))
    self.alphav = nn.Parameter(torch.empty(seq_len ,))
    self.tokenq = nn.Parameter(torch.empty(k_dim , k_dim))
    self.tokenv = nn.Parameter(torch.empty(v_dim , v_dim))

    self._init_parameters()

  def _init_parameters(self):
    nn.init.xavier_uniform_(self.tokenq)
    nn.init.xavier_uniform_(self.tokenv)
    nn.init.constant_(self.alphaq, 0.0)
    nn.init.constant_(self.alphav, 0.0)

  def forward(self , embedding_matrix):
    """
    Makes a forward pass through the attention module, given a 3D embedding matrix of shape
    (batch_size , seq_len , embd_dim)

    Returns tensor of context aware value vectors of shape (batch_size , seq_len , v_dim)
    """
    Q = self.Wq(embedding_matrix) #(batch_size , seq_len , k_dim)
    K = self.Wk(embedding_matrix) #(batch_size , seq_len , k_dim)
    V = self.Wv(embedding_matrix) #(batch_size , seq_len , v_dim)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = self.seq_len

    tokenq = self.tokenq.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , k_dim , k_dim)
    tokenv = self.tokenv.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , v_dim , v_dim)
    alphaq = self.alphaq.view(1,seq_len,1).repeat(batch_size,1,dk) # (batch_size , seq_len , k_dim)
    alphav = self.alphav.view(1,seq_len,1).repeat(batch_size,1,dv) # (batch_size , seq_len , v_dim)
  
    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2))) #(batch_size , k_dim , seq_len)
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2))) #(batch_size , v_dim , seq_len)

    token_temp_q = F.tanh(pdt_q.transpose(1,2)) #(batch_size , seq_len , k_dim)
    token_temp_v = F.tanh(pdt_v.transpose(1,2)) #(batch_size , seq_len , v_dim)

    
    # Computation of dimension aware knobs for better learning of position aware temperature for query vectors

    attn_map_exps = torch.bmm(Q , K.transpose(1,2)) #(batch_size , seq_len , seq_len)
    max_ids = attn_map_exps.argmax(dim=2, keepdim=True)
    # max_ids: (batch_size, seq_len, 1)
    max_ids_exp = max_ids.expand(-1, -1, dk)  # (batch_size, seq_len, dk)

    # Gather the max key for each position and batch
    K_max = torch.gather(K, 1, max_ids_exp)  # (batch_size, seq_len, dk)

    # Sum over sequence dimension
    K_sum = K.sum(dim=1, keepdim=True)  # (batch_size, 1, dk)

    # Compute knobs
    knobs = Q * (K_sum - seq_len * K_max)  # (batch_size, seq_len, dk)
    knobs = knobs.abs()
    # Knob computation ends here

    indicesk = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesk = indicesk.expand(seq_len, dk)  # (seq_len, dk)
    indicesk = indicesk.unsqueeze(0).expand(batch_size, seq_len, dk)  # (batch_size, seq_len, dk)
    
    indicesv = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesv = indicesv.expand(seq_len, dv)  # (seq_len, dv)
    indicesv = indicesv.unsqueeze(0).expand(batch_size, seq_len, dv)  # (batch_size, seq_len, dv)

    position_temp_q = 1 + torch.sigmoid(alphaq - knobs) * torch.log(indicesk) #(batch_size , seq_len ,dk)
    position_temp_v = 1 + torch.sigmoid(alphav) * torch.log(indicesv) #(batch_size , seq_len ,dv)
  
    temps_q = position_temp_q + token_temp_q #(batch_size , seq_len , dk)
    temps_v = position_temp_v + token_temp_v #(batch_size , seq_len , dv)

    gated_Q = temps_q * Q #(batch_size , seq_len , kdim)
    gated_V = temps_v * V #(batch_size , seq_len , vdim)

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return torch.bmm(attention_scores , gated_V)

  @torch.no_grad()
  def attention_map(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns a tensor of attention maps,
    of shape (batch_size , seq_len , seq_len)
    """
    Q = self.Wq(embedding_matrix) #(batch_size , seq_len , k_dim)
    K = self.Wk(embedding_matrix) #(batch_size , seq_len , k_dim)
    V = self.Wv(embedding_matrix) #(batch_size , seq_len , v_dim)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = self.seq_len

    tokenq = self.tokenq.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , k_dim , k_dim)
    tokenv = self.tokenv.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , v_dim , v_dim)
    alphaq = self.alphaq.view(1,seq_len,1).repeat(batch_size,1,dk) # (batch_size , seq_len , k_dim)
    alphav = self.alphav.view(1,seq_len,1).repeat(batch_size,1,dv) # (batch_size , seq_len , v_dim)
  
    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2))) #(batch_size , k_dim , seq_len)
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2))) #(batch_size , v_dim , seq_len)

    token_temp_q = F.tanh(pdt_q.transpose(1,2)) #(batch_size , seq_len , k_dim)
    token_temp_v = F.tanh(pdt_v.transpose(1,2)) #(batch_size , seq_len , v_dim)

    
    # Computation of dimension aware knobs for better learning of position aware temperature for query vectors

    attn_map_exps = torch.bmm(Q , K.transpose(1,2)) #(batch_size , seq_len , seq_len)
    max_ids = attn_map_exps.argmax(dim=2, keepdim=True)
    # max_ids: (batch_size, seq_len, 1)
    max_ids_exp = max_ids.expand(-1, -1, dk)  # (batch_size, seq_len, dk)

    # Gather the max key for each position and batch
    K_max = torch.gather(K, 1, max_ids_exp)  # (batch_size, seq_len, dk)

    # Sum over sequence dimension
    K_sum = K.sum(dim=1, keepdim=True)  # (batch_size, 1, dk)

    # Compute knobs
    knobs = Q * (K_sum - seq_len * K_max)  # (batch_size, seq_len, dk)
    knobs = knobs.abs()
    # Knob computation ends here

    indicesk = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesk = indicesk.expand(seq_len, dk)  # (seq_len, dk)
    indicesk = indicesk.unsqueeze(0).expand(batch_size, seq_len, dk)  # (batch_size, seq_len, dk)
    
    indicesv = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesv = indicesv.expand(seq_len, dv)  # (seq_len, dv)
    indicesv = indicesv.unsqueeze(0).expand(batch_size, seq_len, dv)  # (batch_size, seq_len, dv)

    position_temp_q = 1 + torch.sigmoid(alphaq - knobs) * torch.log(indicesk) #(batch_size , seq_len ,dk)
    position_temp_v = 1 + torch.sigmoid(alphav) * torch.log(indicesv) #(batch_size , seq_len ,dv)
  
    temps_q = position_temp_q + token_temp_q #(batch_size , seq_len , dk)
    temps_v = position_temp_v + token_temp_v #(batch_size , seq_len , dv)

    gated_Q = temps_q * Q #(batch_size , seq_len , kdim)
    gated_V = temps_v * V #(batch_size , seq_len , vdim)

    attention_wts = torch.bmm(gated_Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)
    return attention_scores

  @torch.no_grad()
  def token_aware_temperatures(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns two tensors of shape
    (batch_size , seq_len , query_dim) and (batch_size , seq_len , key_dim) of token aware temperatures correponding to query and value vectors
    """
    Q = self.Wq(embedding_matrix) #(batch_size , seq_len , k_dim)
    K = self.Wk(embedding_matrix) #(batch_size , seq_len , k_dim)
    V = self.Wv(embedding_matrix) #(batch_size , seq_len , v_dim)

    batch_size = embedding_matrix.shape[0]

    tokenq = self.tokenq.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , k_dim , k_dim)
    tokenv = self.tokenv.unsqueeze(0).repeat(batch_size,1,1) #(batch_size , v_dim , v_dim)
    
  
    pdt_q = torch.bmm(tokenq , F.gelu(Q.transpose(1,2))) #(batch_size , k_dim , seq_len)
    pdt_v = torch.bmm(tokenv , F.gelu(V.transpose(1,2))) #(batch_size , v_dim , seq_len)

    token_temp_q = F.tanh(pdt_q.transpose(1,2)) #(batch_size , seq_len , k_dim)
    token_temp_v = F.tanh(pdt_v.transpose(1,2)) #(batch_size , seq_len , v_dim)


    return token_temp_q ,token_temp_v

  @torch.no_grad()
  def position_aware_temperatures(self , embedding_matrix):
    """
    Takes an embedding matrix as input and returns two tensors of shape
    (batch_size , seq_len, query_dim) and (batch_size , seq_len , value_dim) of position aware temperatures correponding to query and value vectors
    """
    Q = self.Wq(embedding_matrix) #(batch_size , seq_len , k_dim)
    K = self.Wk(embedding_matrix) #(batch_size , seq_len , k_dim)
    V = self.Wv(embedding_matrix) #(batch_size , seq_len , v_dim)

    batch_size = embedding_matrix.shape[0]
    dk = K.shape[-1]
    dv = V.shape[-1]
    seq_len = self.seq_len

    alphaq = self.alphaq.view(1,seq_len,1).repeat(batch_size,1,dk) # (batch_size , seq_len , k_dim)
    alphav = self.alphav.view(1,seq_len,1).repeat(batch_size,1,dv) # (batch_size , seq_len , v_dim)
    
    # Computation of dimension aware knobs for better learning of position aware temperature for query vectors

    attn_map_exps = torch.bmm(Q , K.transpose(1,2)) #(batch_size , seq_len , seq_len)
    max_ids = attn_map_exps.argmax(dim=2, keepdim=True)
    # max_ids: (batch_size, seq_len, 1)
    max_ids_exp = max_ids.expand(-1, -1, dk)  # (batch_size, seq_len, dk)

    # Gather the max key for each position and batch
    K_max = torch.gather(K, 1, max_ids_exp)  # (batch_size, seq_len, dk)

    # Sum over sequence dimension
    K_sum = K.sum(dim=1, keepdim=True)  # (batch_size, 1, dk)

    # Compute knobs
    knobs = Q * (K_sum - seq_len * K_max)  # (batch_size, seq_len, dk)
    knobs = knobs.abs()
    # Knob computation ends here

    indicesk = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesk = indicesk.expand(seq_len, dk)  # (seq_len, dk)
    indicesk = indicesk.unsqueeze(0).expand(batch_size, seq_len, dk)  # (batch_size, seq_len, dk)
    
    indicesv = torch.arange(1, seq_len + 1, device=alphaq.device).unsqueeze(1)  # (seq_len, 1)
    indicesv = indicesv.expand(seq_len, dv)  # (seq_len, dv)
    indicesv = indicesv.unsqueeze(0).expand(batch_size, seq_len, dv)  # (batch_size, seq_len, dv)

    position_temp_q = 1 + torch.sigmoid(alphaq - knobs) * torch.log(indicesk) #(batch_size , seq_len ,dk)
    position_temp_v = 1 + torch.sigmoid(alphav) * torch.log(indicesv) #(batch_size , seq_len ,dv)

    return position_temp_q , position_temp_v
