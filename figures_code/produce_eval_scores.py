import torch
from models import *

import data

import training

from tqdm import tqdm

MODELS_FOLDER = "E:/Master/Avhandling/Models/models"

IMPL = "fixslice"
PRED_TARGET = "sbox"
TARGET_BYTE = 0
ARCH = "zhang"
INTERVAL_START = 0
INTERVAL_END = 1000

N_TRACES = 10
EPOCHS = 100


BATCH_SIZE = 200
_, val_loader, _ = data.get_dataloaders(
    BATCH_SIZE,
    PRED_TARGET,
    TARGET_BYTE,
    INTERVAL_START,
    INTERVAL_END,
)

model_name = f"{IMPL}-{PRED_TARGET}-byte{TARGET_BYTE}-{ARCH}-{INTERVAL_START}_{INTERVAL_END}"
with open(f"{MODELS_FOLDER}/eval/{model_name}.txt", 'a') as f:
    f.write(f"{N_TRACES} traces\n")


for epoch in tqdm(range(EPOCHS), desc='evaluating', unit='epoch'):
    epoch = epoch+1

    model = torch.load(f"{MODELS_FOLDER}/{model_name}/epoch{epoch}.pt", weights_only=False)

    if PRED_TARGET == "2sbox":
        score = training.mean_2sbox_rank(model, val_loader, n_traces=N_TRACES)
    elif PRED_TARGET == "sbox":
        score = training.mean_sbox_rank(model, val_loader, n_traces=N_TRACES)


    with open(f"{MODELS_FOLDER}/eval/{model_name}.txt", 'a') as f:
        print(score)
        f.write(f"{score}\n")