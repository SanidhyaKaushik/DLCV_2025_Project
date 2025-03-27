import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AttentionModule(nn.Module):
  def __init__(self , embd_dim , k_dim , v_dim, mask = False):
    super(AttentionModule , self).__init__()
    self.mask = mask
    self.Wq = nn.Linear(embd_dim , k_dim)
    self.Wk = nn.Linear(embd_dim , k_dim)
    self.Wv = nn.Linear(embd_dim , v_dim)

  def forward(self , embedding_matrix): #embedding_matrix is a 3D tensor of shape (batch_size , seq_len , embd_dim)
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    dk = K.shape[-1]
    seq_len = K.shape[1]
    batch_size = embedding_matrix.shape[0]

    attention_wts = torch.bmm(Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return torch.bmm(attention_scores , V)

  def attention_map(self , embedding_matrix):
    Q = self.Wq(embedding_matrix)
    K = self.Wk(embedding_matrix)
    V = self.Wv(embedding_matrix)

    dk = K.shape[-1]
    seq_len = K.shape[1]
    batch_size = embedding_matrix.shape[0]

    attention_wts = torch.bmm(Q , K.transpose(1,2))/(dk**0.5)

    if self.mask:
      mask = torch.triu(torch.ones(seq_len, seq_len , device = attention_wts.device), diagonal=1).bool()
      attention_wts = attention_wts.masked_fill(mask.unsqueeze(0).expand(batch_size , -1 , -1), float('-inf'))

    attention_scores = torch.softmax(attention_wts  ,dim = -1 , dtype = torch.float32)

    return attention_scores