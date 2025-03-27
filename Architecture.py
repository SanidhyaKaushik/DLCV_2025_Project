import torch
import torch.nn as nn
import torch.nn.functional as F
from Decoder_Only_Block import DecoderOnlyBlock

class DecoderOnlyArchitecture(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_len, num_decoder_blocks=6, pad_token_id=0):
        super(DecoderOnlyArchitecture, self).__init__()
        self.embd_dim = embd_dim
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, embd_dim))
        self.decoder_blocks = nn.ModuleList([DecoderOnlyBlock(embd_dim, seq_len) for _ in range(num_decoder_blocks)])
        self.lm_head = nn.Linear(embd_dim, vocab_size)

    def forward(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        
        if seq_len > self.seq_len:
            input_tokens = input_tokens[:, -self.seq_len:]
      
        elif seq_len < self.seq_len:
            pad_length = self.seq_len - seq_len
            input_tokens = torch.cat([torch.full((batch_size, pad_length), self.pad_token_id, device=input_tokens.device), input_tokens], dim=1)
        
        embeddings = self.token_embedding(input_tokens) + self.position_embedding[:, :self.seq_len, :]
        
        decoder_output = embeddings
        for decoder in self.decoder_blocks:
            decoder_output = decoder(decoder_output)
        
        last_token_rep = decoder_output[:, -1, :]
        
        logits = self.lm_head(last_token_rep)
        return logits