"""
Model 4 — Encoder-Decoder with Multiplicative (Luong) Attention.

Luong et al. (2015), "general" scoring variant. Unlike the additive/Bahdanau
attention in Model 3 (which scores alignment with a small feed-forward
network *before* the decoder RNN step), Luong attention runs the decoder GRU
first to get the current hidden state h_t, and then scores every encoder
position with a bilinear ("multiplicative") product

    e_i = h_t^T W_a h_i

softmaxes those scores into attention weights, forms the context vector, and
concatenates it with h_t through a tanh layer to get the final attentional
hidden state used to predict the next word. This is computationally cheaper
than additive attention (one matrix multiply vs. a full MLP per step) while
usually performing comparably.
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
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden  # outputs: (batch, src_len, hidden_dim), hidden: (1, batch, hidden_dim)


class MultiplicativeAttention(nn.Module):
    """Luong 'general' score: e_i = h_t^T W_a h_i."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, decoder_output, encoder_outputs, mask):
        # decoder_output: (batch, 1, hidden_dim), encoder_outputs: (batch, src_len, hidden_dim)
        energy = self.W_a(decoder_output)  # (batch, 1, hidden_dim)
        scores = torch.bmm(energy, encoder_outputs.transpose(1, 2)).squeeze(1)  # (batch, src_len)
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=config.PAD_IDX)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.attention = MultiplicativeAttention(hidden_dim)
        self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input_tok))
        output, hidden = self.gru(embedded, hidden)  # output: (batch, 1, hidden_dim)

        attn_weights = self.attention(output, encoder_outputs, mask)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch, 1, hidden_dim)

        concat_input = torch.cat((output, context), dim=2).squeeze(1)
        attentional_hidden = torch.tanh(self.concat(concat_input))
        logits = self.fc_out(attentional_hidden)
        return logits, hidden, attn_weights


class Seq2Seq(nn.Module):
    name = "multiplicative_attention"
    display_name = "Encoder-Decoder + Multiplicative (Luong) Attention"

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
