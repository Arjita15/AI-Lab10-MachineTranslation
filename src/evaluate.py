"""BLEU evaluation + qualitative sample translations for a trained model."""

import time

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from src import config
from src.dataset import make_loader

_smoother = SmoothingFunction().method4


@torch.no_grad()
def evaluate_bleu(model, pairs, src_vocab, tgt_vocab, device, batch_size=64):
    """Runs greedy decoding over `pairs` and returns corpus BLEU + timing stats."""
    model.eval()
    loader = make_loader(pairs, src_vocab, tgt_vocab, batch_size=batch_size, shuffle=False)

    references, hypotheses = [], []
    n_sentences = 0
    t0 = time.time()
    for src, src_len, tgt, _ in loader:
        src, tgt = src.to(device), tgt.to(device)
        preds = model.translate(src, src_len)  # (batch, max_len)
        for i in range(src.size(0)):
            ref_ids = tgt[i].tolist()
            ref_sentence = tgt_vocab.decode(ref_ids)
            hyp_sentence = tgt_vocab.decode(preds[i].tolist())
            references.append([ref_sentence.split()])
            hypotheses.append(hyp_sentence.split())
        n_sentences += src.size(0)
    elapsed = time.time() - t0

    bleu = corpus_bleu(references, hypotheses, smoothing_function=_smoother)
    return {
        "bleu": bleu,
        "num_sentences": n_sentences,
        "total_seconds": elapsed,
        "sentences_per_second": n_sentences / elapsed if elapsed > 0 else float("inf"),
    }


@torch.no_grad()
def sample_translations(model, pairs, src_vocab, tgt_vocab, device, n=10, seed=config.RANDOM_STATE):
    import random
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))

    results = []
    model.eval()
    for src_sent, tgt_sent in sample:
        src_ids = [config.SOS_IDX] + src_vocab.encode(src_sent) + [config.EOS_IDX]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_len = torch.tensor([len(src_ids)], dtype=torch.long)
        pred_ids = model.translate(src_tensor, src_len)[0].tolist()
        pred_sentence = tgt_vocab.decode(pred_ids)
        results.append({
            "source": src_sent,
            "reference": tgt_sent,
            "prediction": pred_sentence,
        })
    return results
