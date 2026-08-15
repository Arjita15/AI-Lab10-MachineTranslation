"""PyTorch Dataset/DataLoader plumbing for the (english, french) sentence pairs."""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src import config


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_sent, tgt_sent = self.pairs[idx]
        src_ids = [config.SOS_IDX] + self.src_vocab.encode(src_sent) + [config.EOS_IDX]
        tgt_ids = [config.SOS_IDX] + self.tgt_vocab.encode(tgt_sent) + [config.EOS_IDX]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=config.PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=config.PAD_IDX)
    return src_padded, src_lens, tgt_padded, tgt_lens


def make_loader(pairs, src_vocab, tgt_vocab, batch_size=config.BATCH_SIZE, shuffle=True):
    ds = TranslationDataset(pairs, src_vocab, tgt_vocab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
