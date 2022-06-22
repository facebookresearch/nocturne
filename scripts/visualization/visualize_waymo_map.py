# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Plot the text file representation of a protobuf."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pprint

pp = pprint.PrettyPrinter()

data = {}

current = data
file = 'output.txt'
show_tracks = True
parent_keys = []
with open(file, 'r') as f:
    lines = f.read().split('\n')
    for line in lines:
        # print(line)
        if ":" in line:
            k, v = [x.strip() for x in line.split(':')]
            if k in current:
                current[k].append(v)
            else:
                current[k] = [v]
        elif "{" in line:
            k = line[:-1].strip()
            if k not in current:
                current[k] = []
            parent_keys.append(k)
            current[k].append({})
            current = current[k][-1]
        elif "}" in line:
            current = data
            for k in parent_keys[:-1]:
                current = current[k][-1]
            parent_keys = parent_keys[:-1]
        else:
            pass

# message Scenario:
# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/scenario.proto
print('\nScenario')
print(data.keys())

# message Track, message ObjectState:
# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/scenario.proto
print('\nObjects (vehicles, pedestrians, cyclists..)')
print(len(data['tracks']))
print(data['tracks'][0].keys())
print(len(data['tracks'][0]['states']))
print(data['tracks'][0]['states'][0].keys())

# message MapFeature:
# https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/map.proto
print('\nMap (roads, lanes..)')
print(len(data['map_features']))
print(data['map_features'][0].keys())

# supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
fig = plt.figure(figsize=(20, 20))

for mf in data['map_features']:
    k = list(mf.keys())[1]
    assert len(mf[k]) == 1
    v = mf[k][0]

    if k == 'lane':
        xs = []
        ys = []
        for pt in v['polyline']:
            xs.append(float(pt['x'][0]))
            ys.append(float(pt['y'][0]))
        plt.plot(xs, ys, color='cyan', linewidth=1)

    elif k == 'road_line':
        edge_type = v['type'][0]
        # linestyle = 'solid' if edge_type == 'TYPE_ROAD_EDGE_BOUNDARY' else 'dashdot'
        # print(edge_type)

        xs = []
        ys = []
        for pt in v['polyline']:
            xs.append(float(pt['x'][0]))
            ys.append(float(pt['y'][0]))
        plt.plot(xs, ys, color='orange')

    elif k == 'road_edge':
        edge_type = v['type'][0]
        linestyle = 'solid' if edge_type == 'TYPE_ROAD_EDGE_BOUNDARY' else 'dashdot'

        xs = []
        ys = []
        for pt in v['polyline']:
            xs.append(float(pt['x'][0]))
            ys.append(float(pt['y'][0]))
        plt.plot(xs, ys, color='black', linestyle=linestyle)

    elif k == 'stop_sign':
        pos = v['position'][0]
        plt.plot(float(pos['x'][0]), float(pos['y'][0]), 'ro')

    elif k == 'crosswalk':
        xs = []
        ys = []
        for pt in v['polygon']:
            xs.append(float(pt['x'][0]))
            ys.append(float(pt['y'][0]))
        plt.plot(xs, ys, color='purple', linestyle=linestyle)

    elif k == 'speed_bump':
        xs = []
        ys = []
        for pt in v['polygon']:
            xs.append(float(pt['x'][0]))
            ys.append(float(pt['y'][0]))
        plt.plot(xs, ys, color='green', linestyle=linestyle)

    else:
        print('Error with key', k)

if show_tracks:
    img_arr = []

    from celluloid import Camera
    camera = Camera(plt.gcf())
    ax = plt.gca()
    # in range(len(data['tracks'][0]['states'])):
    for i in range(20):
        for object in data['tracks']:
            if object['states'][i]['valid'][0] != 'false':
                plt.scatter(float(object['states'][i]['center_x'][0]),
                            float(object['states'][i]['center_y'][0]),
                            c='blue',
                            s=40)
        # TODO(eugenevinitsky) this is a horrible way of copying over the figure
        lines = list(ax.get_lines())
        for obj in lines:
            plt.plot(obj.get_data()[0], obj.get_data()[1])
        camera.snap()
    animation = camera.animate()
    animation.save('animation.mp4')

patches = []
patches.append(mpatches.Patch(color='cyan', label='lane_center'))
patches.append(mpatches.Patch(color='orange', label='road_line'))
patches.append(mpatches.Patch(color='black', label='road_edge'))
patches.append(mpatches.Patch(color='red', label='stop_sign'))
patches.append(mpatches.Patch(color='purple', label='crosswalk'))
patches.append(mpatches.Patch(color='green', label='speedbump'))
plt.legend(handles=patches)

plt.savefig(file.split('.')[0] + '.png')
