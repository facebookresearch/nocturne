# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for plotting ZSC results."""
import os

import matplotlib.pyplot as plt
import numpy as np


def create_heat_map(file, title, save_path, white_switch):
    """Construct a heatmap of the ZSC results.

    Args:
    ----
        file (str): file path to zsc results
        title (str): title of the plot
        save_path (str): path to save it at
        white_switch (float): if the value is greater than white_switch
            we write the cell text as black. This is just to make
            the plots more readable.
    """
    np_arr = np.load(os.path.join(zsc_path, file))
    np_arr_mean = np.mean(np_arr, axis=-1)

    agent_indices = [f'Agent {i}' for i in range(np_arr.shape[0])]

    fig, ax = plt.subplots()
    ax.imshow(np_arr_mean)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(agent_indices)), labels=agent_indices)
    ax.set_yticks(np.arange(len(agent_indices)), labels=agent_indices)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(agent_indices)):
        for j in range(len(agent_indices)):
            if np_arr_mean[i, j] > white_switch:
                color = 'black'
            else:
                color = 'w'
            ax.text(j,
                    i,
                    f'{np.round(np_arr_mean[i, j], decimals=2)}',
                    ha="center",
                    va="center",
                    color=color)

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(save_path)


def compute_average_change(file):
    """Compare cross play to self play."""
    np_arr = np.load(os.path.join(zsc_path, file))
    np_arr_mean = np.mean(np_arr, axis=-1)
    self_play = np.mean(np.diag(np_arr_mean))
    cross_play = np.mean(
        np_arr_mean[np.where(~np.eye(np_arr_mean.shape[0], dtype=bool))])
    self_play_std = np.std(np.diag(np_arr_mean)) / np.sqrt(
        np_arr_mean.shape[0])
    cross_play_std = np.std(
        np_arr_mean[np.where(~np.eye(np_arr_mean.shape[0], dtype=bool))]
    ) / np.sqrt(np_arr_mean.shape[0]**2 - np_arr_mean.shape[0])
    print(
        f'self play: {self_play} ± {self_play_std}, cross play: {cross_play} ± {cross_play_std}'
    )


if __name__ == '__main__':
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.23/srt_v10/17.02.40/23/srt_v10'
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/4/srt_12'
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/4/srt_12'
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/4/srt_12'
    # 10000 on valid
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/4/srt_12'
    # 10000 on train
    # zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/4/srt_12'
    zsc_path = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.06.01/srt_v27/17.35.33/123/srt_v27'
    create_heat_map('train_zsc_goal.npy',
                    "Cross-play Goal Rate",
                    'cross_play_heat_map.png',
                    white_switch=.8)
    create_heat_map('train_zsc_collision.npy',
                    "Cross-play Collision Rate",
                    'cross_play_collision_map.png',
                    white_switch=0.18)
    compute_average_change('train_zsc_goal.npy')
    compute_average_change('train_zsc_collision.npy')
