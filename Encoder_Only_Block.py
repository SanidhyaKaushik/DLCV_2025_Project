import torch 
import torch.nn as nn
import torch.nn.functional as F 
from Selective_Attention_Module.py import SelectiveAttentionModule

class EncoderOnlyBlock(nn.Module):
  def __init__(self ,embd_dim):
    super(EncoderOnlyBlock , self).__init__()
    self.embd_dim = embd_dim
    self.attention_head1 = SelectiveAttentionModule(embd_dim = embd_dim ,k_dim = embd_dim , v_dim = embd_dim, mask = False)
    self.attention_head2 = SelectiveAttentionModule(embd_dim = embd_dim ,k_dim = embd_dim , v_dim = embd_dim, mask = False)
    self.attention_head3 = SelectiveAttentionModule(embd_dim = embd_dim ,k_dim = embd_dim , v_dim = embd_dim, mask = False)
    self.attention_head4 = SelectiveAttentionModule(embd_dim = embd_dim ,k_dim = embd_dim , v_dim = embd_dim, mask = False)
    self.linear1 = nn.Linear(4*embd_dim , embd_dim)
    self.norm1 = nn.LayerNorm(embd_dim)
    self.linear2 = nn.Linear(embd_dim , embd_dim)
    self.norm2 = nn.LayerNorm(embd_dim)

  def forward(self , embedding_matrix): # Embedding matrix has shape (batch_size, seq_len , embd_dim)
    V1 = self.attention_head1(embedding_matrix)
    V2 = self.attention_head2(embedding_matrix)
    V3 = self.attention_head3(embedding_matrix)
    V4 = self.attention_head4(embedding_matrix)

    concatenated_V = torch.cat([V1 , V2 , V3 , V4] , dim = -1) #(batch_size , seq_len , 4embd_dim)
    Proj_V = self.linear1(concatenated_V) #(batch_size , seq_len , embd_dim)
    normalized_V = self.norm1(embedding_matrix + Proj_V)
    return self.norm2(normalized_V + self.linear2(normalized_V)) #(batch_size , seq_len , embd_dim)
