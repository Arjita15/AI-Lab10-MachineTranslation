"""
Model 3 — Encoder-Decoder with Additive (Bahdanau) Attention.

Bahdanau et al. (2015). Instead of forcing the whole source sentence through
one context vector, the encoder (a bidirectional GRU here) keeps its output
at every source position. At each decoding step the decoder computes an
"additive"/"concat"-style alignment score

    e_i = v^T tanh(W [s_{t-1} ; h_i])

between its previous hidden state s_{t-1} and every encoder output h_i,
turns those scores into attention weights with softmax, and uses the
resulting weighted-average context vector (a different one at every step)
alongside the current input token to predict the next word.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: (batch, src_len, hidden_dim*2)

        # combine the two final direction hidden states into one decoder-sized init state
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)  # (batch, hidden_dim*2)
        hidden_init = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)  # (1, batch, hidden_dim)
        return outputs, hidden_init


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: (1, batch, hidden_dim) -> (batch, src_len, hidden_dim)
        src_len = encoder_outputs.size(1)
        dec_hidden = decoder_hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)  # (batch, src_len)
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.attention = AdditiveAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input_tok))  # (batch, 1, embed_dim)

        attn_weights = self.attention(hidden, encoder_outputs, mask)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim*2)

        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)

        logits = self.fc_out(torch.cat((output, context, embedded), dim=2).squeeze(1))
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    name = "additive_attention"
    display_name = "Encoder-Decoder + Additive (Bahdanau) Attention"

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 embed_dim=config.EMBED_DIM, hidden_dim=config.HIDDEN_DIM,
                 dropout=config.DROPOUT):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    @staticmethod
    def _make_mask(src):
        return (src != config.PAD_IDX)

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO):
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src, src_len)
        mask = self._make_mask(src)
        input_tok = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(input_tok, hidden, encoder_outputs, mask)
            outputs[:, t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1).unsqueeze(1)
            input_tok = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    @torch.no_grad()
    def translate(self, src, src_len, max_len=config.MAX_SENT_LEN + 5, return_attn=False):
        self.eval()
        batch_size = src.size(0)
        encoder_outputs, hidden = self.encoder(src, src_len)
        mask = self._make_mask(src)
        input_tok = torch.full((batch_size, 1), config.SOS_IDX, dtype=torch.long, device=src.device)
        outputs, attns = [], []
        for _ in range(max_len):
            logits, hidden, attn_weights = self.decoder.forward_step(input_tok, hidden, encoder_outputs, mask)
            top1 = logits.argmax(1)
            outputs.append(top1)
            attns.append(attn_weights)
            input_tok = top1.unsqueeze(1)
        result = torch.stack(outputs, dim=1)
        if return_attn:
            return result, torch.stack(attns, dim=1)
        return result
