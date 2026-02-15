import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import get_dataloaders
from models import *
import keyrank_rs

import json


device = torch.device('cuda')

def get_best_epoch(model_name) -> int:
    with open(f"models/eval/{model_name}.txt", 'r') as f:
        lines = f.readlines()[1:] # skip "N traces" line
        lines = [float(line.strip()) for line in lines]
        best_epoch = np.array(lines).argmin()

    return best_epoch.item()

def mean_2sbox_rank_per_trace(model : nn.Module, sbox_test_loader : DataLoader):
    
    total_rank = torch.zeros(500)

    for traces, plaintexts, true_key in sbox_test_loader:

        traces : torch.Tensor = traces.to(device).squeeze()
        plaintexts : torch.Tensor = plaintexts.to(device)

        plaintexts = plaintexts.long().detach().cpu().numpy().squeeze()
        plaintexts : np.ndarray  = plaintexts.transpose((1, 0))


        # 2PT model outputs list of 2 tensors
        sbox_scores : torch.Tensor = model(traces)
        sbox_scores = torch.stack(sbox_scores).detach().cpu().numpy()

        keyscores_both = []
        for scores, pt in zip(sbox_scores, plaintexts):

            numpy_keyscores = keyrank_rs.sbox_scores_to_keyscores_parallel(pt, scores)
            keyscores_both.append(torch.Tensor(numpy_keyscores).to(device))

        keyscores_both = [torch.Tensor(np_ks) for np_ks in keyscores_both]

        for keyscores in keyscores_both:
            # logsum scores before calculating rank
            keyscores = keyscores.softmax(dim=1).log()
            keyscore_acc = torch.zeros(256)

            for idx,keyscore in enumerate(keyscores):
                keyscore_acc += keyscore.cpu()
                ranks = keyscore_acc.argsort(descending=True).argsort()
                total_rank[idx] += ranks[int(true_key)] / 2.0

    mean_ranks = total_rank / len(sbox_test_loader)

    return mean_ranks

def mean_sbox_rank_per_trace(model : nn.Module, sbox_test_loader : DataLoader):
    
    total_rank = torch.zeros(500)

    for traces, plaintexts, true_key in tqdm(sbox_test_loader, 'computing keyrank', leave=False):

        traces : torch.Tensor = traces.to(device)
        plaintexts : torch.Tensor = plaintexts.squeeze().to(device)[..., 0] # only first plaintext for single sbox model


        sbox_scores : torch.Tensor = model(traces.squeeze())

        keyscores = torch.empty(sbox_scores.shape)


        plaintexts = plaintexts.long().detach().cpu().numpy()
        numpy_scores = sbox_scores.detach().cpu().numpy()
            
        numpy_keyscores : np.array = keyrank_rs.sbox_scores_to_keyscores_parallel(plaintexts, numpy_scores)

        keyscores = torch.Tensor(numpy_keyscores)

        # logsum scores before calculating rank
        keyscores = keyscores.softmax(dim=1).log()
        keyscore_acc = torch.zeros(256)

        for idx,keyscore in enumerate(keyscores):
            keyscore_acc += keyscore

            ranks = keyscore_acc.argsort(descending=True).argsort()

            total_rank[idx] += ranks[int(true_key)]

    mean_ranks = total_rank / len(sbox_test_loader)

    return mean_ranks



IMPLEMENTATION = "fixslice"
PREDICTION_TARGET = "sbox"
ARCH = "zhang"
TRACE_START = 0
TRACE_END = 1000



bytes = [
    (0,),
]

for byte, in bytes:
    model_name = f"{IMPLEMENTATION}-{PREDICTION_TARGET}-byte{byte}-{ARCH}-{TRACE_START}_{TRACE_END}"
    
    best_epoch = get_best_epoch(model_name)
    
    model = torch.load(f"models/{model_name}/epoch{best_epoch}.pt", weights_only=False)

    _, _, test_loader = get_dataloaders(200, PREDICTION_TARGET, byte, TRACE_START, TRACE_END)

    if PREDICTION_TARGET == "2sbox":
        mean_ranks = mean_2sbox_rank_per_trace(model, test_loader).tolist()
    elif PREDICTION_TARGET == "sbox":
        mean_ranks = mean_sbox_rank_per_trace(model, test_loader).tolist()

    info = {
        "model_name" : model_name,
        "epoch" : int(best_epoch),
        "per_trace_ranks" : mean_ranks,
    }

    with open(f"models/n_trace_scores/{model_name}.txt", 'w') as f:

        json.dump(info, f, indent=4)
