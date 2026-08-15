"""
Central configuration for the comparative NMT study.

random_state is derived from my roll number ACE080BCT015 (as instructed):
we concatenate the digit groups in the roll number, "080" + "015" -> 80015,
and use that as the fixed seed everywhere (data split, weight init, sampling)
so that results are reproducible.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
ROLL_NUMBER = "ACE080BCT015"
RANDOM_STATE = 80015

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "fra.txt")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TRANSLATIONS_DIR = os.path.join(RESULTS_DIR, "translations")

# ---------------------------------------------------------------------------
# Dataset / language pair
# ---------------------------------------------------------------------------
# Source: English, Target: French (as shipped in fra.txt: "eng<TAB>fra<TAB>attribution")
SRC_LANG = "eng"
TGT_LANG = "fra"

MAX_SENT_LEN = 12          # max words per sentence (after tokenization), keeps training tractable
MIN_SENT_LEN = 2
MIN_WORD_FREQ = 2          # words rarer than this become <unk>
NUM_EXAMPLES = 20000       # size of the subsampled, length-filtered corpus we actually train on

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
PAD_TOKEN, PAD_IDX = "<pad>", 0
SOS_TOKEN, SOS_IDX = "<sos>", 1
EOS_TOKEN, EOS_IDX = "<eos>", 2
UNK_TOKEN, UNK_IDX = "<unk>", 3

# ---------------------------------------------------------------------------
# Model hyperparameters (kept modest so all 4 models train on CPU in reasonable time)
# ---------------------------------------------------------------------------
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
GRAD_CLIP = 1.0
PATIENCE = 3  # early stopping patience (epochs without val-loss improvement)

MODEL_NAMES = [
    "rnn_seq2seq",
    "encoder_decoder",
    "additive_attention",
    "multiplicative_attention",
]

DEVICE = "cpu"  # overridden at runtime if CUDA is available
