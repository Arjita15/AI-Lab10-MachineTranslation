"""Factory for building one of the four seq2seq architectures by name."""

from src.models import (
    rnn_seq2seq,
    encoder_decoder,
    attention_additive,
    attention_multiplicative,
)

_REGISTRY = {
    "rnn_seq2seq": rnn_seq2seq.Seq2Seq,
    "encoder_decoder": encoder_decoder.Seq2Seq,
    "additive_attention": attention_additive.Seq2Seq,
    "multiplicative_attention": attention_multiplicative.Seq2Seq,
}


def build_model(name: str, src_vocab_size: int, tgt_vocab_size: int):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](src_vocab_size, tgt_vocab_size)
