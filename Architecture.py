import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder_Only_Block import EncoderOnlyBlock

class EncoderOnlyArchitecture(nn.Module):
    def __init__(self, n_cats, embd_dim, seq_len, num_encoder_blocks=6, pad_token_id=0):
        super(EncoderOnlyArchitecture, self).__init__()
        self.embd_dim = embd_dim
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, embd_dim))
        self.encoder_blocks = nn.ModuleList([EncoderOnlyBlock(embd_dim) for _ in range(num_encoder_blocks)])
        self.classification_head = nn.Linear(embd_dim, n_cats)

    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        
        if seq_len > self.seq_len:
            input_tokens = input_tokens[:, -self.seq_len:]
      
        elif seq_len < self.seq_len:
            pad_length = self.seq_len - seq_len
            input_tokens = torch.cat([torch.full((batch_size, pad_length), self.pad_token_id, device=input_tokens.device), input_tokens], dim=1)
        
        embeddings = self.token_embedding(input_tokens) + self.position_embedding[:, :self.seq_len, :]
        
        encoder_output = embeddings.clone()
        for encoder in self.decoder_blocks:
            encoder_output = decoder(encoder_output)
        
        last_token_rep = encoder_output[:, -1, :]
        
        logits = self.classfication_head(last_token_rep)
        return logits
