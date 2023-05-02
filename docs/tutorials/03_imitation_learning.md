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

## Imitation Learning with Nocturne

This notebook walks through a basic setup to get started with imitation learning (IL) in Nocturne. 

_Last update: April 2023_

+++

### Introduction

Picture yourself driving down a road at 60 km/h, and suddenly, you spot two people in your peripheral vision. Despite having the right of way, you quickly realize that the pedestrians are in conversation and oblivious to your presence. In a split second, you understand that continuing at this speed could result in an accident. You hit the brakes to decelerate the car and wait for them to notice you.

In recent years, a of variety methods have been explored to develop autonomous vehicles that can safely navigate rich and dynamic traffic scenarios, such as the one just described. In this post, we will focus on an approach called **Imitation Learning** (IL), which aims to teach agents purely by **mimicking human behavior**. The underlying idea is that exposing agents to various traffic situations and the corresponding actions of human drivers can lead to the development of an effective strategy.

<figure>
<center>
<img src='https://drive.google.com/uc?id=1D6vrT2OsCaVcKbao_yg9kZWoYu3_IwiF' width=800'/>
<figcaption></figcaption></center>
</figure>

+++

### Problem description

Our goal is to learn a stochastic policy $\pi: \mathcal{S} \times \mathcal{A} \to [0, 1]$ for every component of our action. We consider a particular form of Imitation Learning here, called **Behavioral Cloning** (BC). We use the collected human driver demonstrations as a target and learn a stochastic policy $\pi(a \mid s, \theta)$ that maps states to actions, where $\theta$ denotes a set of learnable parameters. This policy should output a probability distribution over actions given the state that outputs expert actions with high likelihood.

#### State space

In Nocturne, the state space for a vehicle $i$ at time $t$ has two components: **the ego state** $\mathbf{e}_{t}^{i}$, which holds information about the ego vehicle such as its size and location, and its **observation** $\mathbf{o}^i_t$, which is everything the vehicle can see from its current location and configuration. We also we endow an agent with "memory", which means it may have access to observations from previous timepoints. If the memory is set to 5 for example, the full state for a vehicle $i$ is defined as:
\begin{align*}
    \mathbf{s}^{i}_{t} = (\mathbf{o}^{i}_t, \mathbf{o}^{i}_{t-1}, \mathbf{o}^{i}_{t-2}, \mathbf{o}^{i}_{t-3}, \mathbf{o}^{i}_{t-4}, \mathbf{e}^{i}_t)
\end{align*}
where $\mathbf{o}^{i}_t \in \mathbb{R}^V$, with $V$ determined by the cone angle and view distance for a given car (see the previous tutorial: _Nocturne concepts_) and $\mathbf{e}^i_t \in \mathbb{R}^F$ with $F = 10$.

#### Action space

The full action space of a vehicle corresponds to the vehicle's **acceleration**, the **angle of the steering wheel** and the **head tilt** of the driver. However, at this moment, the expert head tilt actions are not available, and so we only consider the acceleration and steering wheel angle here. In this tutorial we assume a discrete action space for the agent, which is parameterized by the provided action bounds and a number of discretizations. At each timestep, a vehicle takes an action $\mathbf{a}_t \in \mathbb{R}^{A}$ with $A = 2$:
\begin{align}
\mathbf{a}_t = (\text{acceleration} \in [-3, 2] \, m/s^2, \text{steering angle} \in [-.7, .7] \, \text{ rad} \, )
\end{align}

---

> **Note**: Since expert actions may fall between the defined discrete actions, they are mapped to the closest value in the action grid. For example, the experiments in the paper used 6 discrete actions for acceleration by uniformly splitting $[-3, 2] m/s^2$.
<figure>
<center>
<img src='https://drive.google.com/uc?id=1QmxmxHZn2Hwai7qMUWJvJwwOiY0SoeA-' width=650'/>
<figcaption></figcaption></center>
</figure>

---

+++

#### Optimization objective

The objective is to maximize the probability of the agent taking actions that the expert would take in a given state.  A common loss function is the negative log-likelihood of the expert's actions under the agent's policy, which we briefly derive here. We start with the definition of the likelihood of the expert's actions under the agent's policy:
\begin{align*}
     p(\mathbf{a}^\text{expert} \mid \mathbf{s}, \theta) 
     &\overset{\text{i.i.d}}{=} \prod_{k=1}^K p(\mathbf{a}_k^{\text{expert}} \mid \mathbf{s}_k, \theta) \\
     &= \prod_{k=1}^K \prod_{i=1}^A \pi (a^{\text{expert}_i} \mid \mathbf{s}_k, \theta) \\
\end{align*}
where $\mathbf{s}$ and $\mathbf{a}^\text{expert}$ are sequences of states and expert actions, respectively, $\theta$ are the policy parameters, $A$ are the number of actions and $K$ is the batch size. Since maximizing the likelihood is equivalent to minimizing the negative log-likelihood, we can take the negative logarithm of both sides of the equation:
\begin{align*}
    - \log p(\mathbf{a}^{\text{expert}} \mid \mathbf{s}, \theta) = - \sum_{k=1}^K \sum_{i=1}^A \pi (a^{\text{expert}_i} \mid \mathbf{s}_k, \theta)
\end{align*}
Which gives us the following loss function:
\begin{align*}
    \mathcal{L}(\theta) = - \frac{1}{K} \sum_{k=1}^K \sum_{i=1}^A \log \pi(a^{\text{expert}_{i}} \mid \mathbf{s}_k, \theta)
\end{align*}

+++

### Implementation

```{code-cell} ipython3
# General dependencies
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings

# Nocturne dependencies
from examples.imitation_learning import waymo_data_loader 
from examples.imitation_learning.filters import MeanStdFilter
from torch.distributions.categorical import Categorical
from nocturne import Simulation

%config InlineBackend.figure_format = 'svg'
sns.set('notebook', font_scale=1.1, rc={'figure.figsize': (8, 3)})
sns.set_style('ticks', rc={'figure.facecolor': 'none', 'axes.facecolor': 'none'})
warnings.filterwarnings("ignore", category=UserWarning)
```

#### Data

The (mini) dataset used can be downloaded [here](https://www.dropbox.com/sh/8mxue9rdoizen3h/AADGRrHYBb86pZvDnHplDGvXa?dl=0) and consists of 1000 training files and 100 validation files. Place the dataset in a folder of your choosing, unzip the folders inside of it, and change the `DATA_FOLDER` below to point to where you have downloaded it.

```{code-cell} ipython3
DATA_FOLDER = '< your_data_folder >'

train_data_path = Path(f'{DATA_FOLDER}/formatted_json_v2_no_tl_train')
valid_data_path = Path(f'{DATA_FOLDER}/formatted_json_v2_no_tl_valid')
valid_data_paths = list(Path(valid_data_path).glob('tfrecord*.json'))
```

#### Helper functions

Let's first define a couple of helper functions:

```{code-cell} ipython3
def find_nearest_grid_idx(action_grids, actions):
    """
    Convert a batch of actions values to a batch of the nearest action indexes (for discrete actions only).
    credits https://stackoverflow.com/a/46184652/16207351
    Args:
        actions_grids (List[Tensor]): A list of one-dimensional tensors representing the grid of possible
            actions for each dimension of the actions tensor.
        actions (Tensor): A two-dimensional tensor of size (batch_size, num_actions).
    Returns:
        Tensor: A tensor of size (batch_size, num_actions) with the indices of the nearest action in the action grids.
    """
    output = torch.zeros_like(actions)
    for i, action_grid in enumerate(action_grids):
        action = actions[:, i]

        # get indexes where actions would be inserted in action_grid to keep it sorted
        idxs = torch.searchsorted(action_grid, action)

        # if it would be inserted at the end, we're looking at the last action
        idxs[idxs == len(action_grid)] -= 1

        # find indexes where previous index is closer (simple grid has constant sampling intervals)
        idxs[action_grid[idxs] - action > torch.diff(action_grid).mean() * 0.5] -= 1

        # write indexes in output
        output[:, i] = idxs
    return output


def compute_log_prob(action_dists, ground_truth_action, action_grids, reduction='mean', return_indexes=False):
    """Compute the log probability of the expert action for a number of action distributions.
        Losses are averaged over observations for each batch by default.

    Args:
        action_dists (List[Categorical]): Distributions over model actions.
        ground_truth_action (Tensor): Action taken by the expert.
    Returns:
        Tensor of size (num_actions, batch_size) if reduction == 'none' else (num_actions)
    """

    # Find indexes in actions grids whose values are the closest to the ground truth actions
    expert_action_idx = find_nearest_grid_idx(
        action_grids=action_grids, 
        actions=ground_truth_action,
    )

    # Stack log probs of actions indexes wrt. Categorial variables for each action dimension
    log_probs = torch.stack([dist.log_prob(expert_action_idx[:, i]) for i, dist in enumerate(action_dists)])

    if reduction == 'none':
        return (log_probs, expert_action_idx) if return_indexes else log_probs    

    elif reduction == 'sum': 
        agg_log_prob = log_probs.sum(axis=1)
        
    elif reduction == 'mean': 
        agg_log_prob = log_probs.mean(axis=1)
    
    return (agg_log_prob, expert_action_idx) if return_indexes else agg_log_prob


def construct_state(scenario, vehicle, view_dist=80, view_angle=np.radians(180)):
    """Construct the full state for a vehicle.
    Args:
        scenario (nocturne_cpp.Scenario): Simulation at a particular timepoint.
        vehicle (nocturne_cpp.Vehicle): A vehicle object in the simulation.
        view_dist (int): Viewing distance of the vehicle.
        view_angle (int): The view cone angle in radians.
    Returns:
        state (ndarray): The vehicle state.
    """
    ego_state = scenario.ego_state(
        vehicle
    )
    visible_state = scenario.flattened_visible_state(
        vehicle, 
        view_dist=view_dist, 
        view_angle=view_angle
    )
    return np.concatenate((ego_state, visible_state))


def evaluate_agent_in_traffic_scene_(
        path_to_file, scenario_config, num_stacked_states, model,
        num_steps=90, step_size=0.1, invalid_pos=-1e4, warmup_period=10,
        allowed_goal_distance=0.5,
    ):
    """Compute the collision and/or goal rate for a file (traffic scene).
    Args:
        path_to_file (str): Path to the traffic scenario file.
        scenario_config (Dict): Initialization parameters.
        num_stacked_states (int): The memory of an agent.
        model (BehavioralCloningAgent): The model to be evaluated.  
        num_steps (int, optional): Number of steps to take in traffic scenario.
        step_size (float, optional): Size of the steps.
        invalid_pos (int, optional): Check if vehicle is in an invalid location.
        warmup_period (int, optional): We start every episode one second in.
        allowed_goal_distance (float, optional): The distance to the goal position that 
            is considered as successfully reaching the goal.

    Returns: 
        collision_rate_vehicles (ndarray): Ratio of vehicles that collided with another vehicle.
        collision_rate_edges (ndarray): Ratio of vehicles that collided with a road edge.
        reached_goal_rate (ndarray): Ratio of vehicles that reached their goal.
    """
    stacked_state = defaultdict(lambda: None)
    
    # Create simulation from file
    sim = Simulation(str(path_to_file), scenario_config)
    scenario = sim.getScenario()
    vehicles = scenario.getVehicles()
    objects_that_moved = scenario.getObjectsThatMoved()

    # Set all vehicles to expert control mode
    for obj in scenario.getVehicles():
        obj.expert_control = True

    # If a model is given, model will control vehicles that moved
    controlled_vehicles = [obj for obj in vehicles if obj in objects_that_moved]
    for veh in controlled_vehicles: veh.expert_control = False

    # Vehicles to check for collisions on
    objects_to_check = [
        obj for obj in controlled_vehicles if (obj.target_position - obj.position).norm() > 0.5
    ]

    collided_with_vehicle = {obj.id: False for obj in objects_to_check}
    collided_with_edge = {obj.id: False for obj in objects_to_check}
    reached_goal = {obj.id: False for obj in objects_to_check}
    
    # Step through the simulation 
    for time in range(num_steps):      
        for veh in controlled_vehicles:
            
            # Get the state for vehicle at timepoint
            state = construct_state(scenario, veh)

            # Stack state
            if stacked_state[veh.getID()] is None: 
                stacked_state[veh.getID()] = np.zeros(len(state) * num_stacked_states, dtype=state.dtype)
            # Add state to the end and convert to tensor
            stacked_state[veh.getID()] = np.roll(stacked_state[veh.getID()], len(state))
            stacked_state[veh.getID()][:len(state)] = state
            state_tensor = torch.Tensor(stacked_state[veh.getID()]).unsqueeze(0)

            # Pred actions
            actions, _ , _ = model(state_tensor)

            # Set vehicle actions (assuming we don't have head tilt)
            veh.acceleration = actions[0]
            veh.steering = actions[1]
            
        # Step the simulator and check for collision
        sim.step(step_size)

        # Once the warmup period is over                    
        if time > warmup_period:            
            for obj in objects_to_check:
                # Check for collisions
                if not np.isclose(obj.position.x, invalid_pos) and obj.collided: 
                    if int(obj.collision_type) == 1:
                        collided_with_vehicle[obj.id] = True
                    if int(obj.collision_type) == 2:
                        collided_with_edge[obj.id] = True   

                # Check if goal has been reached
                if (obj.target_position - obj.position).norm() < allowed_goal_distance:
                    reached_goal[obj.id] = True
        
    # Average
    collisions_with_vehicles = list(collided_with_vehicle.values())
    collisions_with_edges = list(collided_with_edge.values())
    collision_rate_vehicles = collisions_with_vehicles.count(True) / len(collisions_with_vehicles)
    collision_rate_edges = collisions_with_edges.count(True) / len(collisions_with_edges)

    reached_goal_values = list(reached_goal.values())
    reached_goal_rate = reached_goal_values.count(True) / len(reached_goal_values)

    return collision_rate_vehicles, collision_rate_edges, reached_goal_rate
```

#### Data Loader and setup

To work with the raw traffic files (For details, see tutorial 1: _Data structure_) we create an [iterable-style](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) dataset. The Nocturne simulator allows us iterate through a given traffic scenario fast and access the state and corresponding action for each expert driver, for every timestep. The code below creates an infinite generator that gives us a batch of `state` and `expert_action`'s for an arbitrary vehicle and timepoint in the traffic scene each time its called.

---

> **Note**. `waymo_dataloader.py` defines an iterator that extracts the expert trajectories for every vehicle in a set of traffic scenes. Before returning (state, action) pairs, all timepoints are shuffled to break the temporal structure in the data.

---

```{code-cell} ipython3
dataloader_config = {
    'tmin': 0, # Simulation start time
    'tmax': 90, # Simulation end time
    'view_dist': 80, # Viewing distance
    'view_angle': np.radians(180), # Cone angle is extended to compromise for missing head tilt
    'dt': 0.1, # Simulation step size
    'expert_action_bounds': [[-3, 2], [-.7, .7]], # Bounds for (acceleration, steering)
    'expert_position': False, # Not using positions as actions
    'state_normalization': 100, # Normalization constant
    'n_stacked_states': 5, # The number of memory states
    'file_limit': 1000,
}

scenario_config = {
    'start_time': 0, 
    'allow_non_vehicles': True,
    'spawn_invalid_objects': True,
    'max_visible_road_points': 500, # Number of road points a vehicle can observe
    'sample_every_n': 1,
    'road_edge_first': False,
}

train_config = {
    'batch_size': 512, 
    'num_workers': 0, # Use a single worker
    'hidden_layers': [1025, 256, 128], # Model used in paper
    'action_discretizations': [15, 43], # Number of discretizations (acc, steering)
    'action_bounds': [[-6, 6], [-.7, .7]], # Bounds for (acc, steering)
    'lr': 1e-4,
    'num_epochs': 10, 
    'samples_per_epoch': 50_000, 
}

train_dataset = waymo_data_loader.WaymoDataset(
    data_path=train_data_path,
    file_limit=dataloader_config['file_limit'],
    dataloader_config=dataloader_config,
    scenario_config=scenario_config,
)

train_loader = iter(DataLoader(
    train_dataset,
    batch_size=train_config['batch_size'],
    num_workers=train_config['num_workers'],
    pin_memory=True,
))
```

```{code-cell} ipython3
states, expert_actions = next(train_loader)

states.shape, expert_actions.shape
```

#### Model

Our model takes in a batch of states in a random order, and returns a list of categorical distributions over the actions, the indices of the actions and the set of actions taken for each instance.

<figure>
<center>
<img src='https://drive.google.com/uc?id=1wLYk6sHHxtCwznYG_y8hhTNVokdftTk4' width=900'/>
<figcaption></figcaption></center>
</figure>

```{code-cell} ipython3
class BehavioralCloningAgent(nn.Module):
    """Simple Behavioral Cloning class."""
    def __init__(self, num_inputs, config):
        super(BehavioralCloningAgent, self).__init__()
        self.num_states = num_inputs
        self.hidden_layers = config['hidden_layers']
        self.action_discretizations = config['action_discretizations']
        self.action_bounds = config['action_bounds']
        
        # Create an action space
        self.action_grids = [
            torch.linspace(a_min, a_max, a_count, requires_grad=False)
                for (a_min, a_max), a_count in zip(
                    self.action_bounds, self.action_discretizations)
        ]
        self._build_model()

    def _build_model(self):
        """Build agent MLP"""
        
        # Create neural network model
        self.neural_net = nn.Sequential(
            MeanStdFilter(self.num_states), # Pass states through filter
            nn.Linear(self.num_states, self.hidden_layers[0]),
            nn.Tanh(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_layers[i],
                                self.hidden_layers[i + 1]),
                    nn.Tanh(),
                ) for i in range(len(self.hidden_layers) - 1)
            ],
        )
        
        # Map model representation to discrete action distributions
        pre_head_size = self.hidden_layers[-1]
        self.heads = nn.ModuleList([
            nn.Linear(pre_head_size, discretization)
            for discretization in self.action_discretizations
        ])

    def forward(self, state):
        """Forward pass through the BC model.

            Args:
                state (Tensor): Input tensor representing the state of the environment.

            Returns:
                Tuple[List[Tensor], List[Tensor], List[Categorical]]: A tuple of three lists:
                1. A list of tensors representing the actions to take in response to the input state.
                2. A list of tensors representing the indices of the actions in their corresponding action grids.
                3. A list of Categorical distributions over the actions.
            """

        # Feed state to nn model
        outputs = self.neural_net(state)

        # Get distribution over every action in action types (acc, steering, head tilt)
        action_dists_in_state = [Categorical(logits=head(outputs)) for head in self.heads]

        # Get action indices (here deterministic)
        # Find indexes in actions grids whose values are the closest to the ground truth actions
        actions_idx = [dist.logits.argmax(axis=-1) for dist in action_dists_in_state]
        
        # Get action in action grids
        actions = [
            action_grid[action_idx] for action_grid, action_idx in zip(
                self.action_grids, actions_idx)
        ]
        
        return actions, actions_idx, action_dists_in_state
```

#### Training

We run the Behavioral Cloning experiment for ~600 gradient steps (5 epochs * (50.000 // batch_size) samples per epoch).

```{code-cell} ipython3
batch_iters = train_config['samples_per_epoch'] // train_config['batch_size']
# Logging
loss_log = torch.zeros((batch_iters, train_config['num_epochs']))
indiv_loss_log = torch.zeros((batch_iters, train_config['num_epochs'], len(train_config['action_bounds'])))
indiv_acc_log = torch.zeros((batch_iters, train_config['num_epochs'], len(train_config['action_bounds'])))

# Get state space dimension
states, _ = next(train_loader)
num_states = states.shape[1]

# Build model
model = BehavioralCloningAgent(
    num_inputs=num_states, 
    config=train_config
)
model.train()
optimizer = Adam(model.parameters(), lr=train_config['lr'])

# Train
for epoch in range(train_config['num_epochs']):
    running_loss = 0.0

    for batch in tqdm(range(batch_iters),  unit='batch'):
        
        # Get states and expert actions
        states, expert_actions = next(train_loader)
        
        # Zero param gradients
        optimizer.zero_grad()

        # Forward
        model_actions, model_action_idx, action_dists_in_state = model(states)

        # Compute log probabilities and indices of the expert actions
        log_prob, expert_action_idx = compute_log_prob(
            action_dists=action_dists_in_state,
            ground_truth_action=expert_actions,
            action_grids=model.action_grids,
            reduction='mean',
            return_indexes=True,
        )

        # Compute loss
        loss = -log_prob.sum()

        # Backward 
        loss.backward()

        # Grad clipping
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        
        # Optimize
        optimizer.step()

        # Logging
        accuracy = [
            (model_idx == expert_idx).float().mean(axis=0)
            for model_idx, expert_idx in zip(model_action_idx, expert_action_idx.T)
        ]

        running_loss += loss.item()

        # Log acceleration, steering and heading log probs separately
        loss_log[batch, epoch] = loss.item()
        indiv_loss_log[batch, epoch, 0] = -log_prob[0].item()
        indiv_loss_log[batch, epoch, 1] = -log_prob[1].item()
        indiv_acc_log[batch, epoch, 0] = accuracy[0].item()
        indiv_acc_log[batch, epoch, 1] = accuracy[1].item()

    print(f'epoch = {epoch}, loss = {running_loss/batch_iters:.3f}, steering acc = {indiv_acc_log[:, epoch, 0].mean():.2f}, accel acc = {indiv_acc_log[:, epoch, 1].mean():.2f}, \n')
```

```{code-cell} ipython3
plt.plot(loss_log.flatten())
plt.title('Train loss')
plt.xlabel('Step')
plt.ylabel('Loss')
sns.despine()
```

```{code-cell} ipython3
df = pd.DataFrame(loss_log.flatten().detach().numpy())
df.rolling(window=batch_iters).mean().plot()
plt.xlabel('Step')
plt.ylabel('Loss')
sns.despine()
```

```{code-cell} ipython3
action_titles = ['acceleration', 'steering',]
for i in range(len(action_titles)): plt.plot(indiv_loss_log.flatten(end_dim=1)[:, i].detach().numpy(), label=action_titles[i], alpha=.6)
plt.title('Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1, 1))
sns.despine()
```

```{code-cell} ipython3
for i in range(2): plt.plot(indiv_acc_log[:, i].flatten(end_dim=1).detach().numpy()*100, label=action_titles[i])
plt.title('Accuracy for steering and acceleration')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(bbox_to_anchor=(1, 1))
sns.despine()
```

### Evaluation

We evaluate our model based on two metrics (See also Table 2 in the paper):
- _Collision rate (%)_: The fraction of vehicles that collide with another vehicle or road edge.
  - Collision rate reported in the paper: 38.2 %
- _Goal rate (%)_: The fraction of vehicles that achieve their goal.
  - Goal rate reported in the paper: 25.3 %

#### Observations

We observe the following:
- The average **vehicle <> vehicle collision** ratio is ~43%, meaning that on average, slightly less than half of the vehicles collide with another vehicle during a scene. The table and distribution below show that in a ~11% of the files not a single vehicle collides. The median of across files is 48%.
- The average **vehicle <> road edge collision** is ~58% with a median of ~%56. This is higher than the vehicle <> vehicle collision.
- On average, only 2% of the vehicles reach their **goal position**. In more than 75% of the files, there is not a single vehicle than reaches its goal.

```{code-cell} ipython3
#torch.save(model.state_dict(), 'model_weights.pth')
# # Load model
# model = BehavioralCloningAgent(
#     num_inputs=num_states, 
#     config=train_config,
# )
# model.load_state_dict(torch.load('bc_model.pth'))
```

```{code-cell} ipython3
# Eval mode
model.eval()

collision_rate_veh = np.zeros(len(valid_data_paths))
collision_rate_road_edge = np.zeros(len(valid_data_paths))
goal_rates = np.zeros(len(valid_data_paths))

for file_idx, file in enumerate(tqdm(valid_data_paths)):
    # Compute collision rate for each traffic scene
    collision_rate_veh[file_idx], collision_rate_road_edge[file_idx], goal_rates[file_idx] = evaluate_agent_in_traffic_scene_(
        file, scenario_config, num_stacked_states=5, model=model,
    )
```

```{code-cell} ipython3
df_coll_veh = pd.DataFrame(collision_rate_veh)
df_coll_veh.describe().T
```

```{code-cell} ipython3
plt.title(f'Ratio of vehicle <> vehicle collisions across traffic files | Mean: {collision_rate_veh.mean()*100:.2f}')
plt.hist(collision_rate_veh*100, alpha=.8, bins=30)
plt.xlim([0, 100])
plt.xlabel('Collision rate (%)')
plt.ylabel('Number of files')
plt.axvline(x=collision_rate_veh.mean()*100, color='r')
sns.despine()
```

```{code-cell} ipython3
df_coll_edge = pd.DataFrame(collision_rate_road_edge)
df_coll_edge.describe().T
```

```{code-cell} ipython3
plt.title(f'Ratio of vehicle <> road edge collisions across traffic files | Mean: {collision_rate_road_edge.mean()*100:.2f}')
plt.hist(collision_rate_road_edge*100, alpha=.8, bins=30)
plt.xlim([0, 100])
plt.xlabel('Collision rate (%)')
plt.ylabel('# Traffic scenes')
plt.axvline(x=collision_rate_road_edge.mean()*100, color='r')
sns.despine()
```

```{code-cell} ipython3
plt.title(f'Goal rate across traffic files | Mean: {goal_rates.mean()*100:.2f}')
plt.hist(goal_rates*100, )
plt.xlim([0, 100])
plt.xlabel('Goal rate (%)')
plt.ylabel('# Traffic scenes')
plt.axvline(x=goal_rates.mean()*100, color='r')
sns.despine()
```

```{code-cell} ipython3
df_goal_rate = pd.DataFrame(goal_rates)
df_goal_rate.describe().T
```
