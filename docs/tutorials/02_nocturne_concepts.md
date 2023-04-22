---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: nocturne-research
    language: python
    name: python3
---

## Nocturne concepts

This page introduces the most basic elements of nocturne. You can find further information about these [in Section 3 of the paper](https://arxiv.org/abs/2206.09889).

_Last update: April 2023_

```python
import numpy as np

data_path = 'data/example_scenario.json'
```

### Summary

- Nocturne simulations are discretized traffic scenarios. A scenario is a constructed snapshot of traffic situation at a particular timepoint.
- The state of the vehicle of focus is referred to as the ego state. Each vehicles observes the traffic scene from their own viewpoint and a visible state is constructed by parameterizing the view distance, head angle and cone radius of the driver. The action for each vehicle is a `(1, 3)` tuple with the acceleration, steering and head angle of the vehicle. 
- The step method advances the simulation with a desired step size. By default, the dynamics of vehicles are driven by a kinematic bicycle model. If a vehicle is set to expert-controlled mode, its position, heading, and speed will be updated according to a trajectory created by a human expert.


### Simulation

In Nocturne, a simulation discretizes an existing traffic scenario. At the moment, Nocturne supports traffic scenarios from the Waymo Open Dataset, but can be further extended to work with other driving datasets. 

<figure>
<center>
<img src='https://drive.google.com/uc?id=1nv5Rbyf7ZfdqTdaUduXvEI7ncdkLpDjc' width=650'/>
<figcaption></figcaption>An example of a set of traffic scenario's in Nocturne. Upon initialization, a start time is chosen. After each iteration we take a step in the simulation, which gets us to the next scenario. This is done until we reach the end of the simulation. </center>
</figure>

We show an example of this using `example_scenario.json`, where our traffic data is extracted from the Waymo open motion dataset:

```python
from nocturne import Simulation

scenario_config = {
    'start_time': 0, # When to start the simulation
    'allow_non_vehicles': True, # Whether to include cyclists and pedestrians 
    'max_visible_road_points': 10, # Maximum number of road points for a vehicle
    'max_visible_objects': 10, # Maximum number of road objects for a vehicle
    'max_visible_traffic_lights': 10, # Maximum number of traffic lights in constructed view
    'max_visible_stop_signs': 10, # Maximum number of stop signs in constructed view
}

# Create simulation
sim = Simulation(data_path, scenario_config)
```

### Scenario

A simulation consists of a set of scenarios. A scenario is a snapshot of the traffic scene at a particular timepoint. 

Here is how to create a scenario object:

```python
# Get traffic scenario at timepoint
scenario = sim.getScenario()
```

The `scenario` objects holds information we are interested in. Here are a couple of examples:

```python
# The number of road objects in the scene
len(scenario.getObjects())
```

```python
# The road objects that moved at a particular timepoint
objects_that_moved = scenario.getObjectsThatMoved()

print(f'Total # moving objects: {len(objects_that_moved)}\n')
print(f'Object IDs of moving vehicles: \n {[obj.getID() for obj in objects_that_moved]} ')
```

```python
# Number of road lines
len(scenario.road_lines())
```

```python
scenario.getVehicles()[:5]
```

```python
# No cyclists in this scene
scenario.getCyclists()
```

```python
# Select all moving vehicles that move 
moving_vehicles = [obj for obj in scenario.getVehicles() if obj in objects_that_moved]

print(f'Found {len(moving_vehicles)} moving vehicles in scene: {[vehicle.getID() for vehicle in moving_vehicles]}')
```

#### Ego state

The **ego state** is an array with features that describe the current vehicle. This array holds the following information: 
- 0: length of ego vehicle
- 1: width of ego vehicle
- 2: speed of ego vehicle
- 3: distance to the goal position of ego vehicle
- 4: angle to the goal (target azimuth) 
- 5: desired heading at goal position
- 6: desired speed at goal position
- 7: current acceleration
- 8: current steering position
- 9: current head angle

```python
# Select an arbitrary vehicle
ego_vehicle = moving_vehicles[0]

print(f'Selected vehicle # {ego_vehicle.getID()}')

# Get the state for ego vehicle
scenario.ego_state(ego_vehicle)
```

#### Visible state

We use the ego vehicle state, together with a view distance (how far the vehicle can see) and a view angle to construct the **visible state**. The figure below shows this procedure for a simplified traffic scene. 

Calling `scenario.visible_state()` returns a dictionary with four matrices:
- `stop_signs`: The visible stop signs 
- `traffic_lights`: The states for the traffic lights from the perspective of the ego driver(red, yellow, green).
- `road_points`: The observable road points (static elements in the scene).
- `objects`: The observable road objects (vehicles, pedestrians and cyclists).

<figure>
<center>
<img src='https://drive.google.com/uc?id=1fG43NvPCzaimmW99asRdB73qY-F4u-q0' width='700'/>
<figcaption>To investigate coordination under partial observability, agents in Nocturne can only see an obstructed view of their environment. In this simplified traffic scene, we construct the state for the red ego driver. Note that Nocturne assumes that stop signs can be viewed, even if they are behind another driver. </figcaption></center>
</figure>

\begin{align*}
\end{align*}

<figure>
<center>
<img src='https://drive.google.com/uc?id=1egNkFArE-n4cp6KbeoQyWeePiQ28jYYE' width='300'/>
<figcaption>The same scene, this time showing the view of the yellow car.</figcaption></center>
</figure>


The shape of the visible state is a function of the maximum number of visible objects defined at initialization (traffic lights, stop signs, road objects, and road points) and whether we add padding. If `padding = True`, an array is of size `(max visible objects, # features)` is always constructed, even if there are no visible objects. Otherwise, if `padding = False` new entries are only created when objects are visible.

```python
# Define viewing distance, radius and head angle
view_distance = 80 
view_angle = np.radians(120) 
head_angle = 0
padding = True 

# Construct the visible state for ego vehicle
visible_state = scenario.visible_state(
    ego_vehicle, 
    view_dist=view_distance, 
    view_angle=view_angle,
    head_angle=head_angle,
    padding=padding,
)

visible_state.keys()
```

```python
# There are no visible stop signs at this point
visible_state['stop_signs'].T
```

```python
# Traffic light states are filtered out in this version of Nocturne
visible_state['traffic_lights']
```

```python
# Max visible road points x 13 features
visible_state['road_points'].shape
```

```python
# Number of visible road objects x 13 features 
visible_state['objects'].shape
```

```python
visible_state_dim = sum([val.flatten().shape[0] for key, val in visible_state.items()])

print(f'Dimension flattened visible state: {visible_state_dim}')
```

```python
# We can also flatten the visible state
# flattened has padding: if we miss an object --> zeros
visible_state_flat = scenario.flattened_visible_state(
        ego_vehicle, 
        view_dist=view_distance, 
        view_angle=view_angle, 
        head_angle=head_angle,    
)

visible_state_flat.shape
```

Note that `.flattened_visible_state()` has padding by default.


### Step 

`step(dt)` is a method call on an instance of the Simulation class, where `dt` is a scalar that represents the length of each simulation timestep in seconds. It advances the simulation by one timestep, which can result in changes to the state of the simulation (for example, new positions of objects, updated velocities, etc.) based on the physical laws and rules defined in the simulation.

In the Waymo dataset, the length of the expert data is 9 seconds, a step size of 0.1 is used to discretize each traffic scene. The first second is used as a warm-start, leaving the remaining 8 seconds (80 steps) for the simulation (Details in Section 3.3).

```python
dt = 0.1

# Step the simulation
sim.step(dt)
```

### Vehicle control

By default, vehicles in Nocturne are driven by a **kinematic bicycle model**. This means that calling the `step(dt)` method evolves the dynamics of a vehicle according to the following set of equations (Appendix D in the paper):

\begin{align*}
    \textbf{position: } x_{t+1} &= x_t + \dot{x} \, \Delta t \\
    y_{t+1} &= y_t + \dot{y} \, \Delta t \\
    \textbf{heading: } \theta_{t+1} &= \theta_t + \dot{\theta} \, \Delta t \\ 
    \textbf{speed: } v_{t+1} &= \text{clip}(v_t + \dot{v} \, \Delta t, -v_{\text{max}}, v_{\text{max}}) \\
\end{align*}

with

\begin{align*}
    \dot{v} &= a \\ 
    \bar{v} &= \text{clip}(v_t, + 0.5 \, \dot{v} \, \Delta \, t ,\, - v_{\text{max}}, v_{\text{max}}) \\
    \beta &= \tan^{-1} \left( \frac{l_r \tan (\delta)}{L}  \right) \\
          &= \tan^{-1} (0.5 \tan(\delta)) \\
    \dot{x} &= \bar{v} \cos (\theta + \beta) \\
    \dot{y} &= \bar{v} \sin (\theta + \beta) \\
    \dot{\theta} &= \frac{\bar{v} \cos (\beta)\tan(\delta)}{L}
\end{align*}

where $(x_t, y_t)$ is the position of a vehicle at time $t$, $\theta_t$ is the vehicles heading angle, $a$ is the acceleration and $\delta$ is the steering angle. Finally, $L$ is the length of the car and $l_r = 0.5L$ is the distance to the rear axle of the car.

If we set a vehicle to be **expert-controlled** instead, it will follow the same path as the respective human driver. This means that when we call the `step(dt)` function, the vehicle's position, heading, and speed will be updated to match the next point in the recorded human trajectory.

```python
# By default, all vehicles are not expert controlled
ego_vehicle.expert_control
```

```python
# Set a vehicle to be expert controlled:
ego_vehicle.expert_control = True
```

---

> **Pseudocode**: How `step(dt)` advances the simulation for every vehicle. Full code is implemented in [scenario.cc](https://github.com/facebookresearch/nocturne/blob/ae0a4e361457caf6b7e397675cc86f46161405ed/nocturne/cpp/src/scenario.cc#L264)

---

```Python
for vehicle in vehicles:

    if object is not expert controlled:
        step vehicle dynamics following the kinematic bicycle model
    
    if vehicle is expert controlled:
        get current time & vehicle idx
        vehicle position = expert trajectories[vehicle_idx, time]
        vehicle heading = expert headings[vehicle_idx, time]
        vehicle speed = expert speeds[vehicle_idx, time]
```


### Action space

The action set for a vehicle consists of three components: acceleration, steering and the head angle. Actions are discretized based on a provided upper and lower bound.

The experiments in the paper use:
- 6 discrete actions for **acceleration** uniformly split between $[-3, 2] \, \frac{m}{s^2}$
- 21 discrete actions for **steering** between $[-0.7, 0.7]$ radians 
- 5 discrete actions for **head tilt** between $[-1.6, 1.6]$ radians

This is how you can access an expert action for a vehicle in Nocturne:

```python
# Choose an arbitrary timepoint
time = 5

# Show expert action at timepoint
scenario.expert_action(ego_vehicle, time)
```

```python
type(scenario.expert_action(ego_vehicle, time))
```

```python
# How did the vehicle's position change after taking this action?
scenario.expert_pos_shift(ego_vehicle, time)
```

```python
# How did the head angle change?
scenario.expert_heading_shift(ego_vehicle, time)
```
