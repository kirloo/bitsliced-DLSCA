import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from training import train_model
#from data import get_dataloaders
from tiny_aes_data import get_tinyaes_dataloaders
from models import *

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

EPOCHS = 50
BATCH_SIZE = 200 #
PREDICTION_TARGET = "sbox"
TARGET_BYTE_IDX = 1 # int(sys.argv[1])
LEARNING_RATE = 0.00001

# spiky area
#TRACE_INTERVAL_START = 1150
#TRACE_INTERVAL_END = 1400

TRACE_INTERVAL_START = 1000
TRACE_INTERVAL_END = 2000

INPUT_LENGTH = TRACE_INTERVAL_END - TRACE_INTERVAL_START

if PREDICTION_TARGET in ["2sbox", "2sbox*"]:
    model = CNN_ZHANG_2PT(INPUT_LENGTH)
else: 
    model = CNN_ZHANG_(INPUT_LENGTH)

IMPL = "tinyaes"
ARCH = "zhang"

# Experiment parameters
REGULARIZATION = None
REG_WEIGHT = 0.0
COMBO_LOSS_WEIGHT = 0.5 # Weight of first sbox loss vs second

N_VAL_TRACES = 2

model = model.to(device)


# Zhang 2019
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()


# Experiment metadata
training_metadata = {
    "implementation" : IMPL,
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
    "regularization" : REGULARIZATION,
    "reg_weight" : REG_WEIGHT,
    "combo_loss_weight" : COMBO_LOSS_WEIGHT,
    "validation_N_traces" : N_VAL_TRACES,
    "scores" : ([],[]), # loss, val performance
}


if IMPL in ["zhang", "transnet"]:
    train_loader, val_loader, _ = get_tinyaes_dataloaders(
        BATCH_SIZE,
        PREDICTION_TARGET,
        TARGET_BYTE_IDX,
        TRACE_INTERVAL_START,
        TRACE_INTERVAL_END,
        SEED,
    )
elif IMPL == "tinyaes":
    train_loader, val_loader, _ = get_tinyaes_dataloaders(
        BATCH_SIZE,
        PREDICTION_TARGET,
        TARGET_BYTE_IDX,
        TRACE_INTERVAL_START,
        TRACE_INTERVAL_END,
        SEED,
    )

model_name = f"{IMPL}-{PREDICTION_TARGET}-byte{TARGET_BYTE_IDX}-{ARCH}-{TRACE_INTERVAL_START}_{TRACE_INTERVAL_END}-s{SEED}"

if REGULARIZATION:
    model_name = model_name + f"-{REGULARIZATION}"
if COMBO_LOSS_WEIGHT and COMBO_LOSS_WEIGHT != 0.5:
    model_name = model_name + f"-LW{COMBO_LOSS_WEIGHT}"


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
        REGULARIZATION,
        REG_WEIGHT,
        COMBO_LOSS_WEIGHT,
        validation_N_traces=N_VAL_TRACES,
        implementation=IMPL,
    )

except KeyboardInterrupt:
    print("Cancelled")


import json

with open(f"models/{model_name}/metadata.json", 'w') as f:

    json.dump(training_metadata, f, indent=4)