---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: nocturne-research
  language: python
  name: python3
---

## Data format of a traffic scene

This notebook dives into the data format used to create simulations in Nocturne.

_Last update: April 2023_

```{code-cell} ipython3
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cmap = ['r', 'g', 'b', 'y', 'c'] 
%config InlineBackend.figure_format = 'svg'
sns.set('notebook', font_scale=1.1, rc={'figure.figsize': (8, 3)})
sns.set_style('ticks', rc={'figure.facecolor': 'none', 'axes.facecolor': 'none'})
```

Traffic scenes are constructed by utilizing the Waymo Open Motion dataset. Though every scene is unique, they all have the same basic data structure. 

To load a traffic scene:

```{code-cell} ipython3
# Take an example scene
data_path = 'data/example_scenario.json'

with open(data_path) as file:
    traffic_scene = json.load(file)

traffic_scene.keys()
```

### Global Overview 
A traffic scene consists of:
- `name`: the name of the traffic scenario.
- `objects`: the road objects or moving vehicles in the scene.
- `roads`: the road points in the scene, these are all the stationary objects.
- `tl_states`: the states of the traffic lights, which are filtered out for now. 

```{code-cell} ipython3
traffic_scene['tl_states']
```

```{code-cell} ipython3
traffic_scene['name']
```

```{code-cell} ipython3
pd.Series(
    [
        traffic_scene['objects'][idx]['type']
        for idx in range(len(traffic_scene['objects']))
    ]
).value_counts().plot(kind='bar', rot=45, color=cmap);
plt.title(f'Distribution of road objects in traffic scene. Total # objects: {len(traffic_scene["objects"])}')
plt.show()
```

This traffic scenario only contains vehicles and pedestrians, some scenes have cyclists as well.

```{code-cell} ipython3
pd.Series(
    [
        traffic_scene['roads'][idx]['type']
        for idx in range(len(traffic_scene['roads']))
    ]
).value_counts().plot(kind='bar', rot=45, color=cmap);
plt.title(f'Distribution of road points in traffic scene. Total # points: {len(traffic_scene["roads"])}')
plt.show()
```

### In-Depth: Road Objects

This is a list of different road objects in the traffic scene. For each road object, we have information about its position, velocity, size, in which direction it's heading, whether it's a valid object, the type, and the final position of the vehicle.

```{code-cell} ipython3
# Take the first object
idx = 0

# For each object, we have this information:
traffic_scene['objects'][idx].keys()
```

```{code-cell} ipython3
# Position contains the (x, y) coordinates for the vehicle at every time step
print(json.dumps(traffic_scene['objects'][idx]['position'][:10], indent=4))
```

```{code-cell} ipython3
# Width and length together make the size of the object, and is used to see if there is a collision 
traffic_scene['objects'][idx]['width'], traffic_scene['objects'][idx]['length'] 
```

An object's heading refers to the direction it is pointing or moving in. The default coordinate system in Nocturne is right-handed, where the positive x and y axes point to the right and downwards, respectively. In a right-handed coordinate system, 0 degrees is located on the x-axis and the angle increases counter-clockwise.

Because the scene is created from the viewpoint of an ego driver, there may be instances where the heading of certain vehicles is not available. These cases are represented by the value `-10_000`, to indicate that these steps should be filtered out or are invalid.

```{code-cell} ipython3
# Heading is the direction in which the vehicle is pointing 
plt.plot(traffic_scene['objects'][idx]['heading']);
plt.xlabel('Time step')
plt.ylabel('Heading')
plt.show()
```

```{code-cell} ipython3
# Velocity shows the velocity in the x- and y- directions
print(json.dumps(traffic_scene['objects'][idx]['velocity'][:10], indent=4))
```

```{code-cell} ipython3
# Valid indicates if the state of the vehicle was observed for each timepoint
plt.xlabel('Time step')
plt.ylabel('IS VALID');
plt.plot(traffic_scene['objects'][idx]['valid'], '_', lw=5)
plt.show()
```

```{code-cell} ipython3
# Each object has a goalPosition, an (x, y) position within the scene
traffic_scene['objects'][idx]['goalPosition']
```

```{code-cell} ipython3
# Finally, we have the type of the vehicle
traffic_scene['objects'][idx]['type']
```

### In-Depth: Road Points

Road points are static objects in the scene.

```{code-cell} ipython3
traffic_scene['roads'][idx].keys()
```

```{code-cell} ipython3
# This point represents the edge of a road
traffic_scene['roads'][idx]['type']
```

```{code-cell} ipython3
# Geometry contains the (x, y) position(s) for a road point
# Note that this will be a list for road lanes and edges but a single (x, y) tuple for stop signs and alike
print(json.dumps(traffic_scene['roads'][idx]['geometry'][:10], indent=4));
```
