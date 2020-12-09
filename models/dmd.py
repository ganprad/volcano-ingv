import torch
import torch.nn as nn
import numpy as np
from functools import partial
from scipy.signal import cwt


def build_delay_embeddings(data, embedding_dim,
                           window_size=240000,
                           delay_size=1,
                           wavelet=None,
                           wavelet_scales=None
                           ):
    if wavelet is not None:
        delay_embeddings = list()
    else:
        delay_embeddings = torch.zeros(embedding_dim, window_size)

    size = embedding_dim * delay_size
    for index, stride_idx in enumerate(range(0, size, delay_size)):
        diff = (data.size(0) - window_size)
        if diff < 0:
            d_0 = data[-diff:].unsqueeze(0)
            d_1 = data[:diff].unsqueeze(0)
            d = torch.cat([d_0, d_1], dim=1).squeeze()
            padding = window_size - d.size(0)
            d = nn.functional.pad(d, pad=[0, padding])
            data_window = d[:window_size]
            if wavelet is not None:
                wavelet_emb = torch.tensor(cwt(data_window, wavelet, wavelet_scales))
                delay_embeddings.append(wavelet_emb.unsqueeze(axis=0))
            else:
                delay_embeddings[index, :] = data_window

        elif diff == 0:
            d = data.squeeze()
            data_window = data[:window_size]
            if wavelet is not None:
                wavelet_emb = torch.tensor(cwt(data_window, wavelet, wavelet_scales))
                delay_embeddings.append(wavelet_emb.unsqueeze(axis=0))
            else:
                delay_embeddings[index, :] = data_window

        else:
            d = data[stride_idx:window_size + stride_idx].squeeze()
            data_window = d[:window_size]
            if wavelet is not None:
                wavelet_emb = torch.tensor(cwt(data_window, wavelet, wavelet_scales))
                delay_embeddings.append(wavelet_emb.unsqueeze(axis=0))
            else:
                delay_embeddings[index, :] = data_window

        assert type(d) == torch.Tensor
        assert d.shape != torch.Size([])

    if wavelet is not None:
        delay_embeddings = torch.cat(delay_embeddings, axis=0)

    return delay_embeddings.T


def build_mrhankel(data, embedding_dim, wavelet, layer_stack, delay_size, max_window_size):

    H_stack = []
    for stack_index, s in enumerate(layer_stack):
        ws, n_windows, n_scales = s

        if wavelet is None:
            wavelet_scales = None
            hankel_layer = torch.zeros(ws, embedding_dim, n_windows)
        else:
            wavelet_scales = np.arange(1, n_scales + 1)
            hankel_layer = torch.zeros(ws, wavelet_scales.shape[0], embedding_dim, n_windows)

        for i in range(0, n_windows):
            idx = i * ws
            x = data[idx:idx + ws]
            emb = build_delay_embeddings(x,
                                         embedding_dim=embedding_dim,
                                         wavelet=wavelet,
                                         window_size=ws,
                                         delay_size=delay_size,
                                         wavelet_scales=wavelet_scales)
            hankel_layer[..., i] = emb

        if wavelet is None:
            hankel_layer = hankel_layer.reshape(-1, embedding_dim).T
            padding = max_window_size - hankel_layer.size(-1)
            hankel_layer = nn.functional.pad(hankel_layer, pad=[0, padding])
        else:
            hankel_layer = hankel_layer.reshape(ws * wavelet_scales.shape[0],
                                                n_windows * embedding_dim).T
            padding = max_window_size - hankel_layer.size(-1)
            hankel_layer = nn.functional.pad(hankel_layer, pad=[0, padding])

        H_stack.append(hankel_layer.T)
    return torch.cat(H_stack, dim=-1)


class MultiHankel(nn.Module):
    def __init__(self, embedding_dim, window_size, num_layers, delay_size, wavelet, n_scales_min):
        super(MultiHankel, self).__init__()

        if (window_size // np.power(2, num_layers)) == 1:
            print("Window size of 1. Reduce num_layers")
            raise Exception

        self.layer_stack = [(window_size // np.power(2, i), np.power(2, i), n_scales_min * np.power(2, i))
                            for i in range(0, num_layers)]

        self.build_mrhankel = partial(build_mrhankel,
                                      embedding_dim=embedding_dim,
                                      delay_size=delay_size,
                                      wavelet=wavelet,
                                      layer_stack=self.layer_stack,
                                      max_window_size=window_size)

    def forward(self, x):
        if x.size(1) > 1:
            return torch.stack([self.build_mrhankel(x[:, t]) for t in range(0, x.size(1))])
        else:
            return self.build_mrhankel(x)


class DMD(nn.Module):
    def __init__(self):
        super(DMD, self).__init__()
        self.A_tilde = None
        self.eigenvalues = None
        self.U = None

    def forward(self, h0, h1):
        with torch.no_grad():
            self.U, s, v = torch.svd(h0)
            u_pinv = torch.pinverse(self.U)
            s_inv = torch.diag(torch.reciprocal(s))
            self.A_tilde = torch.einsum('ij,jl,lq,qr', u_pinv, h1, v, s_inv)
            self.eigenvalues, w = torch.eig(self.A_tilde, eigenvectors=True)
            phi = torch.einsum('ij,jk,kl,ls', h1, v, s_inv, w)
            alpha0 = torch.einsum('ij,j', s_inv, v[1, :]).T
            alpha0_pinv = torch.pinverse(alpha0.reshape(-1, 1))
            amplitudes = torch.einsum('ij,jk', w, self.eigenvalues).T * alpha0_pinv
        return phi, amplitudes, v
