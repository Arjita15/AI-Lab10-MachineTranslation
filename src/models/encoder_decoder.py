"""
Model 2 — GRU Encoder-Decoder (Sutskever et al., 2014 style).

Same overall shape as Model 1 (encode everything into one context vector,
decode from it) but the plain RNN cell is swapped for a GRU. The gating
mechanism lets the network learn what to keep/forget across time steps,
which usually gives noticeably better gradient flow and translation quality
than the vanilla RNN, even though it still suffers from the fixed-length
context-vector bottleneck (no attention).
"""

import random

import torch
import torch.nn as nn

from src import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden  # (1, batch, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden):
        embedded = self.dropout(self.embedding(input_tok))
        output, hidden = self.gru(embedded, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    name = "encoder_decoder"
    display_name = "GRU Encoder-Decoder (no attention)"

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM,
                 dropout=config.DROPOUT):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)

        hidden = self.encoder(src, src_len)
        input_tok = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            logits, hidden = self.decoder.forward_step(input_tok, hidden)
            outputs[:, t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1).unsqueeze(1)
            input_tok = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(self, src, src_len, max_len=config.MAX_SENT_LEN + 5):
        self.eval()
        batch_size = src.size(0)
        hidden = self.encoder(src, src_len)
        input_tok = torch.full((batch_size, 1), config.SOS_IDX, dtype=torch.long, device=src.device)
        outputs = []
        for _ in range(max_len):
            logits, hidden = self.decoder.forward_step(input_tok, hidden)
            top1 = logits.argmax(1)
            outputs.append(top1)
            input_tok = top1.unsqueeze(1)
        return torch.stack(outputs, dim=1)
