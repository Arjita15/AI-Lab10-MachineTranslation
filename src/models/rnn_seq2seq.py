"""
Model 1 — Vanilla RNN Seq2Seq.

The simplest of the four architectures: a single-layer *vanilla* RNN (plain
tanh cell, no gating) encodes the source sentence into one final hidden
state, which is then used as the initial hidden state of a second vanilla
RNN that generates the target sentence one token at a time. There is no
attention mechanism — the entire source sentence has to be squeezed into a
single fixed-size vector, which is exactly the bottleneck the other three
models are designed to address.
"""

import random

import torch
import torch.nn as nn

from src import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden  # (1, batch, hidden_dim)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden):
        embedded = self.dropout(self.embedding(input_tok))  # (batch, 1, embed_dim)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


class Seq2Seq(nn.Module):
    name = "rnn_seq2seq"
    display_name = "Vanilla RNN Seq2Seq (no attention)"

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
        input_tok = tgt[:, 0].unsqueeze(1)  # <sos>

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
        return torch.stack(outputs, dim=1)  # (batch, max_len)
