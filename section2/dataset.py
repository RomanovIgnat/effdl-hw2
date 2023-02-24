import torchtext
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset


MAX_LENGTH = 640
MAX_NUM_SENTENCES = 10


def preprocess_data(data_path):
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    data = []
    with open(data_path) as train_data:
        for i, line in tqdm(enumerate(train_data), total=MAX_NUM_SENTENCES):
            if i == MAX_NUM_SENTENCES:
                break
            line = tokenizer(line.strip())[:MAX_LENGTH]
            if len(line) > 0:
                data.append(line)
        vocab = torchtext.vocab.build_vocab_from_iterator(data, specials=['<pad>'])
        vocab.set_default_index(vocab['<unk>'])
        idxs = [vocab(s) for s in data]
    return idxs, vocab


class BrainDataset(Dataset):
    def __init__(self, data_path: str):
        self.idxs, self.vocab = preprocess_data(data_path)
        self.pad_idx = self.vocab['<pad>']

    def __getitem__(self, idx: int):
        return torch.Tensor(self.idxs[idx]).long()

    def __len__(self):
        return len(self.idxs)


class BigBrainDataset(Dataset):
    def __init__(self, data_path: str):
        self.idxs, self.vocab = preprocess_data(data_path)
        self.pad_idx = self.vocab['<pad>']

    def __getitem__(self, idx: int):
        return torch.Tensor(self.idxs[idx]).long()

    def __len__(self):
        return len(self.idxs)


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, n_bins: int = 1):
        self.idxs, self.vocab = preprocess_data(data_path)
        self.pad_idx = self.vocab['<pad>']
        self.n_bins = n_bins

    def __getitem__(self, idx: int):
        return torch.Tensor(self.idxs[idx]).long()

    def __len__(self):
        return len(self.idxs)


class BatchBinSampler(torch.utils.data.Sampler):
    def __init__(self, n_bins, batch_size, data):
        super().__init__(data)
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.data = data
        self.n_iterations = len(data) // self.batch_size
        self.lens = [len(_) for _ in data]
        self.bins_idxs = np.array_split(np.argsort(self.lens), self.n_bins)

    def __iter__(self):
        for _ in range(self.n_iterations):
            cur_bin = np.random.randint(0, self.n_bins)
            yield np.random.choice(self.bins_idxs[cur_bin], self.batch_size, replace=False)

    def __len__(self):
        return self.n_iterations


def collate_fn(batch, pad_value, max_length=None):
    """
    Pad each sequence of the incoming sequences list
    :param pad_value:
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    max_len = max([_.shape[0] for _ in batch]) if max_length is None else max_length
    res = torch.full([len(batch), max_len], fill_value=pad_value)
    for i, toks in enumerate(batch):
        res[i, :toks.shape[0]] = toks
    return res
