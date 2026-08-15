"""
Loads the raw ManyThings.org/Anki English-French file (fra.txt), cleans it,
filters it down to a manageable, length-bounded subset, builds source/target
vocabularies from the training split only, and writes everything needed for
training to data/processed/.

Run directly:  python -m src.preprocess
"""

import pickle
import random
import re
import unicodedata
from collections import Counter

from src import config


def unicode_to_ascii(s: str) -> str:
    """Strip accents, e.g. 'café' -> 'cafe'. We keep this OFF for French by
    default (see normalize_sentence) since accents are meaningful in French,
    but the helper is here for reference / for languages where it helps."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_sentence(s: str, strip_accents: bool = False) -> str:
    s = s.strip().lower()
    if strip_accents:
        s = unicode_to_ascii(s)
    # put a space before punctuation so it becomes its own token, e.g. "it's." -> "it 's ."
    s = re.sub(r"([.!?,])", r" \1", s)
    # collapse anything that isn't a letter (incl. accented French letters) or ./!/?/,/' into a space
    s = re.sub(r"[^a-zA-ZàâäéèêëïîôöùûüçÀÂÄÉÈÊËÏÎÔÖÙÛÜÇ.!?,']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_raw_pairs(path: str):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            eng, fra = parts[0], parts[1]
            pairs.append((normalize_sentence(eng), normalize_sentence(fra)))
    return pairs


def filter_pairs(pairs, min_len=config.MIN_SENT_LEN, max_len=config.MAX_SENT_LEN):
    out = []
    for eng, fra in pairs:
        e_len, f_len = len(eng.split()), len(fra.split())
        if min_len <= e_len <= max_len and min_len <= f_len <= max_len:
            out.append((eng, fra))
    return out


class Vocab:
    def __init__(self, name: str):
        self.name = name
        self.word2idx = {
            config.PAD_TOKEN: config.PAD_IDX,
            config.SOS_TOKEN: config.SOS_IDX,
            config.EOS_TOKEN: config.EOS_IDX,
            config.UNK_TOKEN: config.UNK_IDX,
        }
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.word_freq = Counter()

    def add_sentence(self, sentence: str):
        for word in sentence.split():
            self.word_freq[word] += 1

    def build(self, min_freq: int = config.MIN_WORD_FREQ):
        for word, freq in sorted(self.word_freq.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, sentence: str):
        return [self.word2idx.get(w, config.UNK_IDX) for w in sentence.split()]

    def decode(self, indices):
        words = []
        for i in indices:
            if i == config.EOS_IDX:
                break
            if i in (config.PAD_IDX, config.SOS_IDX):
                continue
            words.append(self.idx2word.get(i, config.UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


def build_and_save(seed: int = config.RANDOM_STATE):
    random.seed(seed)

    print(f"Loading raw pairs from {config.RAW_DATA_PATH} ...")
    pairs = load_raw_pairs(config.RAW_DATA_PATH)
    print(f"  {len(pairs):,} raw pairs")

    pairs = filter_pairs(pairs)
    print(f"  {len(pairs):,} pairs after length filtering "
          f"({config.MIN_SENT_LEN}-{config.MAX_SENT_LEN} words/side)")

    # De-duplicate identical (eng, fra) pairs, keep order deterministic before shuffling
    pairs = list(dict.fromkeys(pairs))
    print(f"  {len(pairs):,} pairs after de-duplication")

    # Subsample to a fixed, tractable corpus size with our seeded RNG
    random.shuffle(pairs)
    pairs = pairs[: config.NUM_EXAMPLES]
    print(f"  {len(pairs):,} pairs kept for the experiment (seed={seed})")

    # Split train/val/test
    n = len(pairs)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * config.VAL_FRAC)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    print(f"  split -> train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # Build vocab from TRAIN split only (avoid leaking val/test tokens into vocab)
    src_vocab = Vocab(config.SRC_LANG)
    tgt_vocab = Vocab(config.TGT_LANG)
    for eng, fra in train_pairs:
        src_vocab.add_sentence(eng)
        tgt_vocab.add_sentence(fra)
    src_vocab.build()
    tgt_vocab.build()
    print(f"  vocab sizes -> {config.SRC_LANG}: {len(src_vocab)}, {config.TGT_LANG}: {len(tgt_vocab)}")

    import os
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    with open(f"{config.PROCESSED_DIR}/data.pkl", "wb") as f:
        pickle.dump(
            {
                "train": train_pairs,
                "val": val_pairs,
                "test": test_pairs,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "seed": seed,
            },
            f,
        )
    print(f"Saved processed data + vocabs to {config.PROCESSED_DIR}/data.pkl")
    return train_pairs, val_pairs, test_pairs, src_vocab, tgt_vocab


if __name__ == "__main__":
    build_and_save()
