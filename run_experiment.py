#!/usr/bin/env python3
"""
Single entry point for the whole experiment.

    python run_experiment.py preprocess          # clean data, build vocab, split, cache to disk
    python run_experiment.py train                # train all 4 models
    python run_experiment.py train --model rnn_seq2seq
    python run_experiment.py evaluate              # BLEU + sample translations + plots for all 4
    python run_experiment.py all                    # preprocess -> train -> evaluate

random_state is fixed to config.RANDOM_STATE (80015, derived from roll no. ACE080BCT015)
everywhere data is shuffled/split, so results are reproducible from a clean checkout.
"""

import argparse
import os
import pickle

import pandas as pd
import torch

from src import config
from src.evaluate import evaluate_bleu, sample_translations
from src.models import build_model
from src.plots import plot_bleu_comparison, plot_loss_curves
from src.preprocess import build_and_save
from src.train import train_model
from src.utils import save_json, set_seed


def load_processed():
    path = os.path.join(config.PROCESSED_DIR, "data.pkl")
    if not os.path.exists(path):
        print("No cached processed data found, running preprocessing first...")
        build_and_save(seed=config.RANDOM_STATE)
    with open(path, "rb") as f:
        return pickle.load(f)


def cmd_preprocess(_args):
    build_and_save(seed=config.RANDOM_STATE)


def cmd_train(args):
    set_seed(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed()

    names = config.MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        set_seed(config.RANDOM_STATE)  # reset seed before each model so runs are comparable
        train_model(name, data["train"], data["val"], data["src_vocab"], data["tgt_vocab"], device)


def cmd_evaluate(_args):
    set_seed(config.RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_processed()
    src_vocab, tgt_vocab = data["src_vocab"], data["tgt_vocab"]

    rows = []
    for name in config.MODEL_NAMES:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"skip {name}: no checkpoint found, train it first")
            continue

        model = build_model(name, len(src_vocab), len(tgt_vocab)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        metrics = evaluate_bleu(model, data["test"], src_vocab, tgt_vocab, device)
        samples = sample_translations(model, data["test"], src_vocab, tgt_vocab, device, n=15)

        print(f"{model.display_name:45s} BLEU={metrics['bleu']*100:5.2f}  "
              f"({metrics['sentences_per_second']:.1f} sent/s)")

        save_json(samples, os.path.join(config.TRANSLATIONS_DIR, f"{name}_samples.json"))
        rows.append({
            "model": name,
            "display_name": model.display_name,
            "bleu": metrics["bleu"] * 100,
            "test_sentences_per_second": metrics["sentences_per_second"],
        })

    df = pd.DataFrame(rows).sort_values("bleu", ascending=False)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    df.to_csv(os.path.join(config.METRICS_DIR, "comparison.csv"), index=False)
    print("\nSaved comparison table to results/metrics/comparison.csv")
    print(df.to_string(index=False))

    plot_loss_curves()
    plot_bleu_comparison()


def cmd_all(args):
    cmd_preprocess(args)
    args.model = "all"
    cmd_train(args)
    cmd_evaluate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess").set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train")
    p_train.add_argument("--model", default="all", choices=config.MODEL_NAMES + ["all"])
    p_train.set_defaults(func=cmd_train)

    sub.add_parser("evaluate").set_defaults(func=cmd_evaluate)
    sub.add_parser("all").set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)
