import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

stats_dict = np.load('perf_stats_1000_new.pkl', allow_pickle=True)
import ipdb; ipdb.set_trace()
plt.figure(dpi=300)
plt.plot(stats_dict['single_avg_fps'][0:50])
plt.xlabel('Number of non-agent vehicles in the scene')
plt.ylabel('Steps-per-second')
plt.savefig('sa_sps.png')

plt.figure(dpi=300)
plt.plot(np.linspace(1, 49, 50), stats_dict['multi_avg_sec_by_agt'][0:50])
plt.xlabel('Number of controlled agents')
plt.ylabel('Time per step (s)')
plt.savefig('ma_time_per_step.png')
