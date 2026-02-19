import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from training import train_model
from data import get_dataloaders
from models import *

from transnet import Transformer

import os

device = torch.device("cuda")


torch.use_deterministic_algorithms(True)

# Variable must be set to allow deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 777

torch_rng = torch.manual_seed(SEED)
np_rng = np.random.default_rng(SEED)

np.random.seed(SEED)

import sys

EPOCHS = 100
BATCH_SIZE = 200 #
PREDICTION_TARGET = "sbox"
TARGET_BYTE_IDX = 1
LEARNING_RATE = 0.00001

TRACE_INTERVAL_START = 900
TRACE_INTERVAL_END = 1200

INPUT_LENGTH = TRACE_INTERVAL_END - TRACE_INTERVAL_START

model = CNN_ZHANG_(INPUT_LENGTH)
model = Transformer(1,8,4,2,16,0.,0.,256,11,2,
                    clamp_len=300,
                    untie_r=True,
                    smooth_pos_emb=False,
                    untie_pos_emb=True
)

model = model.to(device)


# Zhang 2019
optimizer = optim.Adam(model.parameters(), lr=0.00001)
loss_function = nn.CrossEntropyLoss()




# Experiment metadata
training_metadata = {
    "epochs" : EPOCHS,
    "batch_size" : BATCH_SIZE,
    "target_variable" : PREDICTION_TARGET,
    "target_byte_index" : TARGET_BYTE_IDX,
    "learning_rate" : LEARNING_RATE,
    "trace_interval_start" : TRACE_INTERVAL_START,
    "trace_interval_end" : TRACE_INTERVAL_END,
    "seed" : SEED,
    "model" : str(type(model)),
    "optimizer" : str(type(optimizer)),
    "loss" : str(type(loss_function)),
    "scores" : ([],[]),
}



train_loader, val_loader, _ = get_dataloaders(
    BATCH_SIZE,
    PREDICTION_TARGET,
    TARGET_BYTE_IDX,
    TRACE_INTERVAL_START,
    TRACE_INTERVAL_END
)

model_name = f"fixslice-{PREDICTION_TARGET}-byte{TARGET_BYTE_IDX}-zhang-{TRACE_INTERVAL_START}_{TRACE_INTERVAL_END}"

try:
    train_model(
        model,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        model_name,
        training_metadata["scores"],
        EPOCHS,
        PREDICTION_TARGET,
    )

except KeyboardInterrupt:
    print("Cancelled")


import json

with open(f"models/{model_name}/metadata.json", 'w') as f:

    json.dump(training_metadata, f, indent=4)