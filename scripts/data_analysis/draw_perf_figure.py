import pickle
import numpy as np

import matplotlib.pyplot as plt

FILE_NAMES = [
    "./perf_data/perf_stats_2000-1.pkl",
    "./perf_data/perf_stats_2000-2.pkl",
    "./perf_data/perf_stats_2000-3.pkl",
    "./perf_data/perf_stats_2000-4.pkl",
    "./perf_data/perf_stats_2000-5.pkl",
]

NUM_MAX_VEHICLES = 400
X_RANGE = 51

n = len(FILE_NAMES)

single_avg_fps = np.zeros(n)
single_avg_fps_by_veh = np.zeros((NUM_MAX_VEHICLES, n))

multi_avg_fps = np.zeros(n)
multi_avg_sec = np.zeros(n)
multi_avg_fps_by_agt = np.zeros((NUM_MAX_VEHICLES, n))
multi_avg_sec_by_agt = np.zeros((NUM_MAX_VEHICLES, n))
multi_all_fps = np.zeros(n)

multi_avg_agt_num = np.zeros(n)

for i, file in enumerate(FILE_NAMES):
    with open(file, "rb") as f:
        stats = pickle.load(f)

    cur_single_avg_fps = np.sum(stats["single_cnt_by_veh"]) / np.sum(
        stats["single_sec_by_veh"])
    single_avg_fps[i] = cur_single_avg_fps
    single_avg_fps_by_veh[:, i] = stats["single_avg_fps"]

    cur_multi_avg_fps = np.sum(stats["multi_cnt_by_agt"]) / np.sum(
        stats["multi_sec_by_agt"])
    cur_multi_avg_sec = np.sum(stats["multi_sec_by_agt"]) / np.sum(
        stats["multi_cnt_by_agt"])
    multi_avg_fps[i] = cur_multi_avg_fps
    multi_avg_sec[i] = cur_multi_avg_sec
    multi_avg_fps_by_agt[:, i] = stats["multi_avg_fps_by_agt"]
    multi_avg_sec_by_agt[:, i] = stats["multi_avg_sec_by_agt"]
    multi_all_fps[i] = cur_multi_avg_fps * stats["multi_avg_agt_num"]
    multi_avg_agt_num[i] = stats["multi_avg_agt_num"]

single_avg_fps_by_veh_med = np.median(single_avg_fps_by_veh, axis=1)
single_avg_fps_by_veh_min = np.min(single_avg_fps_by_veh, axis=1)
single_avg_fps_by_veh_max = np.max(single_avg_fps_by_veh, axis=1)

single_avg_fps_by_veh_avg = np.mean(single_avg_fps_by_veh, axis=1)
single_avg_fps_by_veh_std = np.std(single_avg_fps_by_veh, axis=1, ddof=1)

print(f"[single] avg_fps = {single_avg_fps}")
print(f"[single] avg_fps = {np.mean(single_avg_fps)} " +
      f"[{np.std(single_avg_fps, ddof=1)}]")

vehicles = [10, 20, 30]
for v in vehicles:
    print(f"[single] vehicle = {v}, fps = {single_avg_fps_by_veh_avg[v]}, " +
          f"std = {single_avg_fps_by_veh_std[v]}")

x = np.arange(X_RANGE)
m = X_RANGE

fig, ax = plt.subplots()
ax.plot(x, single_avg_fps_by_veh_avg[:m])
ax.fill_between(
    x,
    single_avg_fps_by_veh_avg[:m] - single_avg_fps_by_veh_std[:m],
    single_avg_fps_by_veh_avg[:m] + single_avg_fps_by_veh_std[:m],
    alpha=0.25,
)
plt.xlabel("Number of non-agent vehicles")
plt.ylabel("Average steps-per-second")
# plt.show()
plt.savefig("figure_7.png", dpi=300)
plt.clf()

print(f"[multi] avg_agent_num = {multi_avg_agt_num}")
print(f"[multi] avg_agent_num = {np.mean(multi_avg_agt_num)} " +
      f"[{np.std(multi_avg_agt_num, ddof=1)}]")
print(f"[multi] avg_fps = {multi_avg_fps}")
print(f"[multi] avg_fps = {np.mean(multi_avg_fps)} " +
      f"[{np.std(multi_avg_fps, ddof=1)}]")
print(f"[multi] all_fps = {multi_all_fps}")
print(f"[multi] all_fps = {np.mean(multi_all_fps)} " +
      f"[{np.std(multi_all_fps, ddof=1)}]")

multi_avg_fps_by_agt_med = np.median(multi_avg_fps_by_agt, axis=1)
multi_avg_fps_by_agt_min = np.min(multi_avg_fps_by_agt, axis=1)
multi_avg_fps_by_agt_max = np.max(multi_avg_fps_by_agt, axis=1)

multi_avg_sec_by_agt_med = np.median(multi_avg_sec_by_agt, axis=1)
multi_avg_sec_by_agt_min = np.min(multi_avg_sec_by_agt, axis=1)
multi_avg_sec_by_agt_max = np.max(multi_avg_sec_by_agt, axis=1)

multi_avg_fps_by_agt_avg = np.mean(multi_avg_fps_by_agt, axis=1)
multi_avg_fps_by_agt_std = np.std(multi_avg_fps_by_agt, axis=1, ddof=1)

multi_avg_sec_by_agt_avg = np.mean(multi_avg_sec_by_agt, axis=1)
multi_avg_sec_by_agt_std = np.std(multi_avg_sec_by_agt, axis=1, ddof=1)

x = np.arange(1, X_RANGE)
m = X_RANGE - 1
# print(multi_avg_sec_by_agt[:m, :])

fig, ax = plt.subplots()
ax.plot(x, multi_avg_sec_by_agt_avg[:m] * 1000)
ax.fill_between(
    x,
    (multi_avg_sec_by_agt_avg[:m] - multi_avg_sec_by_agt_std[:m]) * 1000,
    (multi_avg_sec_by_agt_avg[:m] + multi_avg_sec_by_agt_std[:m]) * 1000,
    alpha=0.25,
)

plt.xlabel("Number of controlled agents")
plt.ylabel("Average time per step (ms)")
# plt.show()
plt.savefig("figure_8.png", dpi=300)
plt.clf()
