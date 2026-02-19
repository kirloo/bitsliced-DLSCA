import h5py
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset, random_split

from keyrank_rs import sbox_key_permutations

DATAFOLDER = "C:/Data"

train_hdf = h5py.File(f"{DATAFOLDER}/simpleserial-aes-fix-500000-diff-profile.hdf5")
val_test_hdf = h5py.File(f"{DATAFOLDER}/simpleserial-aes-fix-500-diff.hdf5")

train_traces = torch.Tensor(np.array(train_hdf['trace']))
train_plaintexts = torch.Tensor(np.array(train_hdf['data']))
train_keys = torch.Tensor(np.array(train_hdf['key']))

val_test_traces = torch.Tensor(np.array(val_test_hdf['trace']))
val_test_plaintexts = torch.Tensor(np.array(val_test_hdf['data']))
val_test_keys = torch.Tensor(np.array(val_test_hdf['key']))


s_box = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
]

inverted_s_box = [0] * 256

for idx, bytevalue in enumerate(s_box):
    inverted_s_box[bytevalue] = idx



train_sbox_output = torch.empty((500000,2,16))
val_test_sbox_output = torch.empty((1000,500,2,16))

for idx, (key, plaintext) in enumerate(zip(train_keys, train_plaintexts)):

    pt = plaintext.clone()
    block1 = pt[:16]
    block2 = pt[16:]

    block1.map_(key, lambda x,y: s_box[int(x) ^ int(y)])
    block2.map_(key, lambda x,y: s_box[int(x) ^ int(y)])

    block1, block2 = block1.long(), block2.long()

    train_sbox_output[idx, 0] = block1
    train_sbox_output[idx, 1] = block2

# Outer loop is each of only 1000 different keys in testing set
for key_idx in range(val_test_traces.shape[0]):
    for idx, (key, plaintext) in enumerate(zip(val_test_keys[key_idx], val_test_plaintexts[key_idx])):

        pt = plaintext.clone()
        block1 = pt[:16]
        block2 = pt[16:]

        block1.map_(key, lambda x,y: s_box[int(x) ^ int(y)])
        block2.map_(key, lambda x,y: s_box[int(x) ^ int(y)])

        block1,block2 = block1.long(),block2.long()

        val_test_sbox_output[key_idx, idx, 0] = block1
        val_test_sbox_output[key_idx, idx, 1] = block2

# Reshape plaintext set to match stored sbox output
val_test_plaintexts = val_test_plaintexts.reshape(1000, 500, 2, 16)


class TrainingTraceSet(Dataset):
    def __init__(self, traces : torch.Tensor, keys, plaintexts, subkey_idx):

        self.traces = traces
        self.keys = keys[..., subkey_idx]

        # split into individual plaintexts
        plaintexts_split = plaintexts.reshape([plaintexts.shape[0], 2, 16])
        plaintext_bytes = plaintexts_split[..., subkey_idx]
        self.plaintexts = plaintext_bytes


    def __len__(self):
        return self.traces.shape[0]
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.traces[idx]
        target = self.keys[idx]
        pt_bytes = self.plaintexts[idx]
        return (sample, target, pt_bytes)
    

class TestingTraceSet(Dataset):
    def __init__(self, traces, keys, subkey_idx):

        self.traces = traces
        self.keys = keys[..., subkey_idx]

    def __len__(self):
        return self.traces.shape[0]
    
    def __getitem__(self, idx):
        # One sample, N traces
        traces = self.traces[idx]
        key = self.keys[..., idx]

        return (traces, key)
    

class SboxTestingTraceSet(Dataset):
    def __init__(self, traces, plaintexts, keys, subbyte_idx):

        self.traces = traces

        # Plaintext must be included to map possible sbox outpouts back to keys
        self.plaintexts = plaintexts[..., subbyte_idx]
        self.keys = keys[..., subbyte_idx].squeeze()

    def __len__(self):
        return self.traces.shape[0]
    
    def __getitem__(self, idx):
        # One sample, N traces, N plaintexts, single key
        traces = self.traces[idx]
        plaintexts = self.plaintexts[idx].squeeze()
        key = self.keys[idx]

        return (traces, plaintexts, key)



def trace_sample() -> torch.Tensor:
    return train_traces[44534]



def get_dataloaders(
        batch_size : int,
        prediction_target : str,
        target_byte_idx : int,
        trace_interval_start : int,
        trace_interval_end : int,
        seed = 777,
    ) -> tuple[DataLoader,DataLoader,DataLoader]:

    torch_rng = torch.manual_seed(seed)

    train_traces_trunc = train_traces[..., trace_interval_start:trace_interval_end]
    val_test_traces_trunc = val_test_traces[..., trace_interval_start:trace_interval_end]

    train_traces_mean = train_traces_trunc.mean()
    train_traces_std = train_traces_trunc.std()

    train_traces_norm = (train_traces_trunc - train_traces_mean) / train_traces_std
    val_test_traces_norm = (val_test_traces_trunc - train_traces_mean) / train_traces_std

    key_train_set = TrainingTraceSet(train_traces_norm[:,:], train_keys, train_plaintexts, target_byte_idx)
    key_val_test_set = TestingTraceSet(val_test_traces_norm, val_test_keys, target_byte_idx)
    key_val_set, key_test_set = random_split(key_val_test_set, [0.5, 0.5], generator = torch_rng)

    sbox_train_set = TrainingTraceSet(train_traces_norm[:,:], train_sbox_output, train_plaintexts, target_byte_idx)
    sbox_val_test_set = SboxTestingTraceSet(val_test_traces_norm, val_test_plaintexts, val_test_keys, target_byte_idx)
    sbox_val_set, sbox_test_set = random_split(sbox_val_test_set, [0.5, 0.5], generator = torch_rng)

    if prediction_target in ["key", "2sbox*"]: # When mapping before loss computation we also need key for loss
        train_loader = DataLoader(key_train_set, batch_size=batch_size, shuffle=True, generator=torch_rng)
    else:
        train_loader = DataLoader(sbox_train_set, batch_size=batch_size, shuffle=True, generator=torch_rng)

    if prediction_target == "key":
        val_loader = DataLoader(key_val_set, shuffle=False)
        test_loader = DataLoader(key_test_set, shuffle=False)
    else:
        val_loader = DataLoader(sbox_val_set, shuffle=False)
        test_loader = DataLoader(sbox_test_set, shuffle=False)


    return train_loader,val_loader,test_loader