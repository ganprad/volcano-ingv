# This script generates two files

from models.dmd import MultiHankel, DMD
from scipy.signal import gaussian
import torch
import numpy as np
import pandas as pd

# Params:
DATA_PATH = "data/train"
TRAIN = "data/train.csv"
PREPROC = "data/train_preprocessed"

# Hyperparameters:
EMBEDDING_DIM = 25
WINDOW_SIZE = 100
NUM_LAYERS = 3
DELAY_SIZE = 10
OVERLAP = 1

if __name__ == "__main__":
    train = pd.read_csv(f"{TRAIN}")
    random_file = train.iloc[np.random.choice(train.index)]
    data = {random_file.segment_id:
            pd.read_csv(f"{DATA_PATH}/{str(random_file.segment_id)}.csv")}
    data_seg = data[random_file.segment_id].fillna(0.0)

    tensor_data = torch.tensor(data_seg.values)
    hankel = MultiHankel(embedding_dim=EMBEDDING_DIM,
                         window_size=WINDOW_SIZE,
                         num_layers=NUM_LAYERS,
                         delay_size=DELAY_SIZE,
                         wavelet=gaussian,
                         n_scales_min=2)

    x0 = tensor_data[:-OVERLAP, :]
    x1 = tensor_data[OVERLAP:, :]
    h0 = hankel(x0)
    h1 = hankel(x1)
    rh0 = h0.squeeze()
    rh0 = rh0.reshape(rh0.size(0) * WINDOW_SIZE, rh0.size(2))
    rh1 = h1.squeeze()
    rh1 = rh1.reshape(rh1.size(0) * WINDOW_SIZE, rh1.size(2))
    dmd = DMD()
    phi, amplitudes, delay_coords = dmd(rh0, rh1)

    pd.DataFrame(phi.numpy()).to_csv(f"{PREPROC}/phi/{str(random_file.segment_id)}.csv")
    pd.DataFrame(delay_coords.numpy()).to_csv(f"{PREPROC}/delay_embeddings/{str(random_file.segment_id)}.csv")
    pd.DataFrame(amplitudes.numpy()).to_csv(f"{PREPROC}/amplitudes/{str(random_file.segment_id)}.csv")

