import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import os

import keyrank_rs

device = torch.device("cuda")



def mean_keyrank(model : nn.Module, test_loader : DataLoader):
    """Compute mean keyrank across trace set of N keys with M traces per key"""
    total_rank = 0

    for traces, key in test_loader:

        traces = traces.to(device)
        key = key.to(device)

        output : torch.Tensor = model(traces.squeeze())

        output = output.softmax(dim=0).log().sum(dim=0)

        ranks = output.argsort(dim=-1, descending=True).argsort(dim=-1)

        total_rank += ranks[..., int(key)].mean(dtype=float)

    mean_rank = total_rank / len(test_loader)

    return mean_rank


def mean_sbox_rank(model, sbox_test_loader, n_traces=500, plaintext=0):
    """Compute mean keyrank from an sbox predicting model across trace set of N keys with M traces per key"""
    total_rank = 0

    for traces, plaintexts, true_key in sbox_test_loader:

        traces : torch.Tensor = traces.to(device)

        plaintexts : torch.Tensor = plaintexts[..., plaintext].squeeze().to(device)

        sbox_scores : torch.Tensor = model(traces.squeeze()[0:n_traces, :])

        plaintexts = plaintexts.long().detach().cpu().numpy()[0:n_traces]
        numpy_scores = sbox_scores.detach().cpu().numpy()
            
        numpy_keyscores = keyrank_rs.sbox_scores_to_keyscores_parallel(plaintexts, numpy_scores)

        keyscores = torch.Tensor(numpy_keyscores).to(device)

        # logsum scores before calculating rank
        keyscores = keyscores.softmax(dim=1).log().sum(dim=0)
        #keyscores = keyscores.mean(dim=0).softmax(dim=0)

        ranks = keyscores.argsort(dim=-1, descending=True).argsort(dim=-1)
        
        total_rank += ranks[..., int(true_key)].mean(dtype=float)

    mean_rank = total_rank / len(sbox_test_loader)

    return mean_rank

def mean_2sbox_rank(model : nn.Module, sbox_test_loader, n_traces=500):
    
    total_rank = 0

    for traces, plaintexts, true_key in tqdm(sbox_test_loader, desc='evaluating'):

        traces : torch.Tensor = traces.to(device).squeeze()[0:n_traces]
        plaintexts : torch.Tensor = plaintexts.to(device)

        plaintexts = plaintexts.long().detach().cpu().numpy().squeeze()[0:n_traces]
        plaintexts : np.ndarray  = plaintexts.transpose((1, 0))


        # 2PT model outputs list of 2 tensors
        sbox_scores : torch.Tensor = model(traces)
        sbox_scores = torch.stack(sbox_scores).detach().cpu().numpy()


        for scores, pt in zip(sbox_scores, plaintexts):

            numpy_keyscores = keyrank_rs.sbox_scores_to_keyscores_parallel(pt, scores)
            keyscores = torch.Tensor(numpy_keyscores).to(device)

            # logsum scores before calculating rank
            keyscores = keyscores.softmax(dim=1).log().sum(dim=0)

            rank = keyscores.argsort(dim=-1, descending=True).argsort(dim=-1)

            total_rank += rank[int(true_key)].mean(dtype=float)


    mean_rank = total_rank / (2 * len(sbox_test_loader))

    return mean_rank


def single_sbox_loss(loss_fn, output, targets, sbox=0):
    return loss_fn(output, targets[..., sbox])

def individual_score_loss(loss_fn, outputs, targets):
    "Combine loss from two outputs and targets"
    loss1 = loss_fn(outputs[0], targets[..., 0])
    loss2 = loss_fn(outputs[1], targets[..., 1])
    loss = (loss1 + loss2) * 0.5
    return loss

def combined_score_loss(loss_fn, outputs, target, plaintexts):
    # map both sbox scores to key score
    # and combine (plain sum?) before computing loss

    # target must be key
    plaintexts1 = plaintexts[..., 0]
    plaintexts2 = plaintexts[..., 1]

    perms1 = torch.Tensor(keyrank_rs.sbox_key_permutations(plaintexts1.long().tolist())).long().to(device)
    perms2 = torch.Tensor(keyrank_rs.sbox_key_permutations(plaintexts2.long().tolist())).long().to(device)

    mapped1 = outputs[0].gather(-1, perms1)
    mapped2 = outputs[1].gather(-1, perms2)

    # Should we do something here? TODO
    # Seems like we need to average, larger logits == bad ??
    output_keyscores = (mapped1 + mapped2) * 0.5

    loss = loss_fn(output_keyscores, target)

    return loss
                


def train_model(
        model : nn.Module,
        optimizer : optim.Optimizer,
        loss_fn,
        train_loader : DataLoader,
        val_loader : DataLoader,
        folder : str,
        scores : tuple[list,list] = ([],[]),
        n_epochs = 10,
        prediction_target = "key",
    ):

    model.to(device)

    os.makedirs(f"models/{folder}", exist_ok=True)

    train_losses = scores[0]
    val_ranks = scores[1]

    for epoch in range(n_epochs):

        model.train()

        train_loss = 0

        for input, target, plaintexts in tqdm(train_loader, desc='training', unit='batch', leave=True):

            target = target.type(torch.LongTensor)

            # Move batch to GPU
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(input)

            if prediction_target == "2sbox*": # map before loss computation
                loss = combined_score_loss(loss_fn, output, target, plaintexts)
            elif prediction_target == "2sbox":
                loss = individual_score_loss(loss_fn, output, target)
            elif prediction_target == "sbox":
                loss = single_sbox_loss(loss_fn, output, target)
            elif prediction_target == "sbox2":
                loss = single_sbox_loss(loss_fn, output, target, sbox=1)
            
            loss.backward()

            optimizer.step()

            train_loss+=loss.item()


        train_loss /= len(train_loader)

        train_losses.append(float(train_loss))


        # Validate
        model.eval()
        with torch.no_grad():
            if prediction_target == "key":
                val_mean_rank = mean_keyrank(model, val_loader, 10)
            elif prediction_target == "sbox":
                val_mean_rank = mean_sbox_rank(model, val_loader, 10)
            elif prediction_target == "sbox2":
                val_mean_rank = mean_sbox_rank(model, val_loader, 10, plaintext=1)
            elif prediction_target in ["2sbox", "2sbox*"]:
                val_mean_rank = mean_2sbox_rank(model, val_loader, 10)
            
        val_ranks.append(float(val_mean_rank))

        torch.save(model, f"models/{folder}/epoch{epoch}.pt")
        
        print(f"Epoch #{epoch+1} of {n_epochs}, training loss: {train_loss:.3f}, val mean keyrank: {val_mean_rank:.3f}")

    return (train_losses, val_ranks)