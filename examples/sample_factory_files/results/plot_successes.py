# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Util for plotting eval_sample_factory.py output."""
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    plt.figure()
    num_arr = np.load('success_by_veh_number.npy')
    for i in range(num_arr.shape[0]):
        veh_num_arr = num_arr[i, i]
        plt.figure()
        plt.plot(list(range(len(veh_num_arr))), veh_num_arr[:, 0])
        plt.plot(list(range(len(veh_num_arr))), veh_num_arr[:, 1])
        plt.plot(list(range(len(veh_num_arr))),
                 veh_num_arr[:, 1] + veh_num_arr[:, 0])
        plt.xlabel('num vehicles')
        plt.ylabel('rate')
        plt.legend(['goal rate', 'collide rate', 'sum'])
        plt.title('goal rate as function of number of vehicles')
        plt.savefig(f'{i}_goal_func_num.png')
        plt.close()
    num_arr = np.load('success_by_dist.npy')
    for i in range(num_arr.shape[0]):
        dist_arr = num_arr[i, i]
        plt.figure()
        plt.plot(10 * np.array(list(range(len(dist_arr)))), dist_arr[:, 0])
        plt.plot(10 * np.array(list(range(len(dist_arr)))), dist_arr[:, 1])
        plt.plot(10 * np.array(list(range(len(dist_arr)))),
                 dist_arr[:, 1] + dist_arr[:, 0])
        plt.xlabel('distance')
        plt.ylabel('rate')
        plt.legend(['goal rate', 'collide rate', 'sum'])
        plt.title('goal rate as function of start distance')
        plt.savefig(f'{i}_goal_func_dist.png')
        plt.close()
