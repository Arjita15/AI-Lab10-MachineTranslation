#!/usr/bin/env python3
"""
Quick interactive/CLI demo: translate an English sentence into French using
any of the four trained models.

    python translate.py --model additive_attention "how are you"
    python translate.py --model rnn_seq2seq --interactive
"""

import argparse
import pickle

import torch

from src import config
from src.models import build_model
from src.preprocess import normalize_sentence


def load_everything(model_name, device):
    with open(f"{config.PROCESSED_DIR}/data.pkl", "rb") as f:
        data = pickle.load(f)
    src_vocab, tgt_vocab = data["src_vocab"], data["tgt_vocab"]

    model = build_model(model_name, len(src_vocab), len(tgt_vocab)).to(device)
    ckpt_path = f"{config.CHECKPOINT_DIR}/{model_name}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, src_vocab, tgt_vocab


@torch.no_grad()
def translate_sentence(sentence, model, src_vocab, tgt_vocab, device):
    normalized = normalize_sentence(sentence)
    src_ids = [config.SOS_IDX] + src_vocab.encode(normalized) + [config.EOS_IDX]
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(src_ids)], dtype=torch.long)
    pred_ids = model.translate(src_tensor, src_len)[0].tolist()
    return tgt_vocab.decode(pred_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", nargs="?", default=None, help="English sentence to translate")
    parser.add_argument("--model", default="additive_attention", choices=config.MODEL_NAMES)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab = load_everything(args.model, device)
    print(f"Loaded {model.display_name}\n")

    if args.interactive:
        print("Type an English sentence (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            print(" ", translate_sentence(line, model, src_vocab, tgt_vocab, device))
    elif args.sentence:
        print(translate_sentence(args.sentence, model, src_vocab, tgt_vocab, device))
    else:
        parser.print_help()
