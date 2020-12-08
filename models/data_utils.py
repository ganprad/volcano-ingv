import matplotlib.pyplot as plt
import numpy as np
import scaleogram
import pandas as pd


def plot_sensors(data_seg):
    def trim_axs(axs, N):
        """little helper to massage the axs list to have correct length..."""
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]

    f, axs = plt.subplots(5, 2, figsize=(12, 12), constrained_layout=True, sharex=True)
    axs = trim_axs(axs, len(data_seg.columns))
    for ax, case in zip(axs, data_seg.columns):
        sensor_number = case.split("_")[-1]
        ax.set_title(f'sensor:{sensor_number}')
        ax.plot(data_seg[case].values, marker='o', ls='-', ms=4)
    plt.show()


def plot_sensors_cwt(data_seg, n_periods, wavelet, figsize=(12, 32)):
    f, axs = plt.subplots(10, 2, figsize=figsize,
                          constrained_layout=True,
                          sharex=True)
    periods = np.arange(1, n_periods)
    scales = scaleogram.periods2scales(periods)
    time = data_seg.reset_index()['index'].values
    for ax, case in zip(axs, data_seg.columns):
        sensor_number = case.split("_")[-1]
        ax[0].set_title(f'Time series: sensor:{sensor_number}')
        ax[0].plot(data_seg[case].values, marker='o', ls='-', ms=0.05, alpha=0.5)

        ax[1] = scaleogram.cws(time=time,
                               wavelet=wavelet,
                               scales=scales,
                               signal=data_seg[case].values,
                               coikw={"alpha": 0.9}, ax=ax[1])
        ax[1].set_title(f'CWT: sensor:{sensor_number}: wavelet: {wavelet}')
    plt.show()


def get_time_series(q, data_dict):
    while not q.empty():
        work = q.get()
        data_dict[work[0]] = pd.read_csv(f"../data/train/{str(work[1])}.csv").to_dict()
    q.task_done()
    return True

