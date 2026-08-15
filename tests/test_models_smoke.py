"""
Lightweight smoke tests: build each of the 4 models with random dummy vocab
sizes and make sure a forward pass and greedy-decode pass both run and
produce tensors of the expected shape. This does NOT check translation
quality (that needs real training) — it only guards against shape bugs,
so a broken model fails fast instead of silently training garbage for
20 minutes.

Run with:  python -m pytest tests/ -v
       or:  python tests/test_models_smoke.py
"""

import torch

from src import config
from src.models import build_model

BATCH, SRC_LEN, TGT_LEN = 4, 7, 6
SRC_VOCAB, TGT_VOCAB = 50, 60


def _dummy_batch():
    src = torch.randint(4, SRC_VOCAB, (BATCH, SRC_LEN))
    src[:, 0] = config.SOS_IDX
    src_len = torch.full((BATCH,), SRC_LEN, dtype=torch.long)
    tgt = torch.randint(4, TGT_VOCAB, (BATCH, TGT_LEN))
    tgt[:, 0] = config.SOS_IDX
    return src, src_len, tgt


def test_forward_and_translate_shapes():
    for name in config.MODEL_NAMES:
        model = build_model(name, SRC_VOCAB, TGT_VOCAB)
        src, src_len, tgt = _dummy_batch()

        out = model(src, src_len, tgt)
        assert out.shape == (BATCH, TGT_LEN, TGT_VOCAB), f"{name}: bad forward() shape {out.shape}"

        preds = model.translate(src, src_len, max_len=5)
        assert preds.shape == (BATCH, 5), f"{name}: bad translate() shape {preds.shape}"
        print(f"OK  {name:28s} forward={tuple(out.shape)}  translate={tuple(preds.shape)}")


if __name__ == "__main__":
    test_forward_and_translate_shapes()
    print("\nAll smoke tests passed.")
