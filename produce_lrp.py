import captum
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule, PropagationRule

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import *
from data import get_dataloaders, s_box
import torch

import numpy as np

def get_best_epoch(model_name) -> int:
    with open(f"models/eval/{model_name}.txt", 'r') as f:
        lines = f.readlines()[1:] # skip "N traces" line
        lines = [float(line.strip()) for line in lines]
        best_epoch = np.array(lines).argmin()

    return best_epoch.item()

IMPLEMENTATION = "fixslice"
PREDICTION_TARGET = "key"
TARGET_BYTE = 1
ARCH = "zhang"

TRACE_START = 0
TRACE_END = 1000

INPUT_LENGTH = TRACE_END - TRACE_START

model_name = f"{IMPLEMENTATION}-{PREDICTION_TARGET}-byte{TARGET_BYTE}-{ARCH}-{TRACE_START}_{TRACE_END}"

epoch = 63


device = torch.device("cuda")

sbox_model = torch.load(f"models/{model_name}/epoch{epoch}.pt")
sbox_model.to(device)

_, _, test_loader = get_dataloaders(
    200,
    PREDICTION_TARGET,
    TARGET_BYTE,
    TRACE_START,
    TRACE_END,
)



layer_lrp = captum.attr.LRP(sbox_model)

total_attr = torch.zeros(INPUT_LENGTH)
total_abs_attr = torch.zeros(INPUT_LENGTH)

#for trace, plaintexts, key in tqdm(test_loader):
for trace, key in tqdm(test_loader):
    for mod in sbox_model.modules():
        mod.rule = EpsilonRule()

    trace = trace[:, 0, :]
    #plaintexts = plaintexts[:, 0, :].squeeze()
    key = key.item()

    # Add round key (plain key) then sbox
    #sbox1 = s_box[int(plaintexts[0]) ^ int(key)]
    #sbox2 = s_box[int(plaintexts[1]) ^ int(key)]

    if PREDICTION_TARGET == "sbox":
        attr = layer_lrp.attribute(trace.to(device), target=sbox1).squeeze()
    elif PREDICTION_TARGET == "sbox2":
        attr = layer_lrp.attribute(trace.to(device), target=sbox2).squeeze()
    elif PREDICTION_TARGET == "key":
        attr = layer_lrp.attribute(trace.to(device), target=int(key)).squeeze()
    
    np_attr = attr.squeeze().detach().cpu().numpy()
    np_abs_attr = attr.abs().squeeze().detach().cpu().numpy()

    total_attr += np_attr
    total_abs_attr += np_abs_attr

avg_attr = total_attr / len(test_loader)
avg_abs_attr = total_abs_attr / len(test_loader)


data = {
    "avg_attr" : avg_attr.tolist(),
    "avg_abs_attr" : avg_abs_attr.tolist(),
}



import json

with open(f"models/lrp/{model_name}.json", 'w') as f:

    json.dump(data, f, indent=4)