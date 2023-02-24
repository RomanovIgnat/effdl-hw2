import time
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from section2.dataset import MAX_LENGTH
from section2.transformer import PositionalEncoding, generate_square_subsequent_mask


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_heads=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            PositionalEncoding(d_model, max_len=MAX_LENGTH),
        )

        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.encoder(x)
        padding_mask = generate_square_subsequent_mask(x.shape[1]).to(x.device)

        res = self.decoder(
            tgt=emb,
            memory=emb,
            tgt_mask=padding_mask,
            memory_mask=padding_mask
        )

        return self.classifier(res)


def get_gpt2_model(vocab_size):
    return GPTLikeModel(vocab_size)


def run_epoch(dataloader, model):
    device = 'cuda'
    times = []
    for x in tqdm(dataloader):
        start = time.time()
        x = x.to(device)
        _ = model(x)

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    return {
        'min': np.min(times),
        'max': np.max(times),
        'mean': np.mean(times),
        'median': np.median(times),
    }
