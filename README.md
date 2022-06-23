# Nocturne

Nocturne is a 2D driving simulator, built in C++ for speed and exported as a Python library.

It is currently designed to handle traffic scenarios from the [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset), and with some work could be extended to support different driving datasets. Using the Python library `nocturne`, one is able to train controllers for AVs to solve various tasks from the Waymo dataset, which we provide as a benchmark, then use the tools we offer to evaluate the designed controllers.

Using this rich data source, Nocturne contains a wide range of scenarios whose solution requires the formation of complex coordination, theory of mind, and handling of partial observability. Below we show replays of the expert data, scentered on the light blue agent, with the corresponding view of the agent on the right.
<!-- <p float="left" align="center">
  <img src="https://user-images.githubusercontent.com/33672752/174244303-91fb597a-0d3e-4a92-8901-46e1134c28b4.gif" width="250" height="250"/>
  <img src="https://user-images.githubusercontent.com/33672752/174244860-65865e95-0592-4279-ab5d-f40842092cc7.gif" width="250" height="250"/>
  <img src="https://user-images.githubusercontent.com/33672752/174244327-51f98409-4afd-424e-88f5-29892e89d796.gif" width="250" height="250"/>
</p> -->
![Intersection Scene with Obscured View](./docs/readme_files/git_intersection_combined.gif)

Nocturne features a rich variety of scenes, ranging from parking lots, to merges, to roundabouts, to unsignalized intersections.

![Intersection Scene with Obscured View](./docs/readme_files/nocturne_3_by_3_scenes.gif)

The corresponding paper is available at: [https://arxiv.org/abs/2206.09889](https://arxiv.org/abs/2206.09889). Please cite the paper and not the GitHub repository, using the following citation:

```bibtex
@article{nocturne2022,
  author  = {Vinitsky, Eugene and Lichtlé, Nathan and Yang, Xiaomeng and Amos, Brandon and Foerster, Jakob},
  journal = {arXiv preprint arXiv:2206.09889},
  title   = {{Nocturne: a scalable driving benchmark for bringing multi-agent learning one step closer to the real world}},
  url     = {https://arxiv.org/abs/2206.09889},
  year    = {2022}
}
```

# Installation

## Dependencies

[CMake](https://cmake.org/) is required to compile the C++ library. 

Run `cmake --version` to see whether CMake is already installed in your environment. If not, refer to the CMake website instructions for installation, or you can use:

- `sudo apt-get -y install cmake` (Linux)
- `brew install cmake` (MacOS)

Nocturne uses [SFML](https://github.com/SFML/SFML) for drawing and visualization, as well as on [pybind11](https://pybind11.readthedocs.io/en/latest/) for compiling the C++ code as a Python library.

To install SFML:

- `sudo apt-get install libsfml-dev` (Linux)
- `brew install sfml` (MacOS)

pybind11 is included as a submodule and will be installed in the next step.

## Installing Nocturne

Start by cloning the repo:

```bash
git clone https://github.com/nathanlct/nocturne.git
cd nocturne
```

Then run the following to install git submodules:

```bash
git submodule sync
git submodule update --init --recursive
```

If you are using [Conda](https://docs.conda.io/en/latest/) (recommended), you can instantiate an environment and install Nocturne into it with the following:

```bash
# create the environment and install the dependencies
conda env create -f environment.yml

# activate the environment where the Python library should be installed
conda activate nocturne

# run the C++ build and install Nocturne into the simulation environment
python setup.py develop
```

If you are not using Conda, simply run the last command to build and install Nocturne at your default Python path.

You should then be all set to use that library from your Python executable:

```python
> python
Python 3.8.11
>>> from nocturne import Simulation
>>> sim = Simulation()
>>> sim.reset()
Resetting simulation.
```

Python tests can be ran with `pytest`.

<details>
<summary><b>Click here for a list of common installation errors</b></summary>

### pybind11 installation errors

If you are getting errors with pybind11, install it directly in your conda environment (eg. `conda install -c conda-forge pybind11` or `pip install pybind11`, cf. https://pybind11.readthedocs.io/en/latest/installing.html for more info).
</details>

## Dataset

### Downloading the dataset
Two versions of the dataset are available:
- a mini-one that is about 1 GB and consists of 1000 training files and 100 validation / test files at: [Link](https://drive.google.com/drive/folders/1URK27v78gKAVirvUahaXK_pT2KeJkBM3?usp=sharing).
- the full dataset (150 GB) and consists of 134453 training files and 12205 validation / test files: [Dropbox Link](https://www.dropbox.com/sh/wv75pjd8phxizj3/AABfNPWfjQdoTWvdVxsAjUL_a?dl=0)

Place the dataset at a folder of your choosing, unzip the folders inside of it, and change the DATA_FOLDER in ```cfgs/config.py``` to point to where you have
downloaded it.

### Constructing the Dataset
If you do want to rebuild the dataset, download the Waymo Motion version 1.1 files.
- Open ```cfgs/config.py``` and change ```DATA_FOLDER``` to be the path to your Waymo motion files
- Run ```python scripts/json_generation/run_waymo_constructor.py --parallel --no_tl --all_files --datatype train valid```. This will construct, in parallel, a dataset of all the train and validation files in the waymo motion data. It should take on the order of 5 minutes with 20 cpus. If you want to include traffic lights scenes, remove the ```--no_tl``` flag.
- To ensure that only files that have a guaranteed solution are included (for example, that there are no files where the agent goal is across an apparently uncrossable road edge), run ```python scripts/json_generation/make_solvable_files.py --datatype train valid```.

## C++ build instructions

If you want to build the C++ library independently of the Python one, run the following:

```bash
cd nocturne/cpp
mkdir build
cd build
cmake ..
make
make install
```

Subsequently, the C++ tests can be ran with `./tests/nocturne_test` from within the `nocturne/cpp/build` directory.


# Usage 
To get a sense of available functionality in Nocturne, we have provided a few examples  in the `examples` folder of how to construct the env (`create_env.py`), how to construct particular observations (`nocturne_functions.py`), and how to render results (`rendering.py`).

The following goes over how to use training algorithms using the Nocturne environment.

## Running the RL algorithms
Nocturne comes shipped with a default Gym environment in ```nocturne/envs/base_env.py```. Atop this we build integration for a few popular RL libraries.

Nocturne by default comes with support for three versions of Proximal Policy Optimization:
1. Sample Factory, a high throughput asynchronous PPO implementation (https://github.com/alex-petrenko/sample-factory)
2. RLlib's PPO (https://github.com/ray-project/ray/tree/master/rllib)
3. Multi-Agent PPO from (https://github.com/marlbenchmark/on-policy)
Each algorithm is in its corresponding folder in examples and has a corresponding config file in cfgs/

*Warning:* only the sample factory code has been extensively swept and tested. The default hyperparameters in there
should work for training the agents from the corresponding paper. The other versions are provided for convenience
but are not guaranteed to train a performant agent with the current hyperparameter settings.

### Important hyperparameters to be aware of
There are a few key hyperparameters that we expect users to care quite a bit about. Each of these can be toggled by adding
```++<hyperparam_name>=<hyperparam_value>``` to the run command.
- ```num_files```: this controls how many training scenarios are used. Set to -1 to use all of them.
- ```max_num_vehicles```: this controls the maximum number of controllable agents in a scenario. If there are more than ```max_num_vehicles``` controllable agents in the scene, we sample ```max_num_vehicles``` randomly from them and set the remainder to be experts. If you want to ensure that all agents are controllable, simply pick a large number like 100.

### Running sample factory
Files from sample factory can be run from examples/sample_factory_files and should work by default by running
```python examples/sample_factory_files/visualize_sample_factory.py algorithm=APPO```
Additional config options for hyperparameters can be found in the config file.

Once you have a trained checkpoint, you can visualize the results and make a movie of them by running ```python examples/sample_factory_files/run_sample_factory.py <PATH TO OUTPUT MODEL>```.

*Warning*: because of how the algorithm is configured, sample-factory works best with a fixed number of agents
operating on a fixed horizon. To enable this, we use the config parameter ```max_num_vehicles``` which initializes the environment with only scenes that have fewer controllable agents than ```max_num_vehicles```. Additionally, if there are fewer than ```max_num_vehicles``` in the scene we add dummy agents that receive a vector of -1 at all timesteps. When a vehicle exits the scene we continue providing it a vector of -1 as an observation and a reward of 0.

### Running RLlib
Files from RLlib examples can be run from examples/rllib_files and should work by default by running
```python examples/rllib_files/run_rllib.py```

### Running on-policy PPO
Files from [MAPPO](https://github.com/marlbenchmark/on-policy) examples can be run from examples/rllib_files and should work by default by running
```python examples/on_policy_files/nocturne_runner.py algorithm=ppo```

## Running the IL Algorithms
Nocturne comes with a baseline implementation of behavioral cloning and a corresponding
DataLoader. This can be run via ```python examples/imitation_learning/train.py```.

# Contributors

<table>
<tbody>
<tr>
<td align="center">
  <a href="https://github.com/eugenevinitsky">
    <img src="https://avatars.githubusercontent.com/u/7660397?v=4" width="100px;" alt="Eugene Vinitsky" style="border-radius: 50%" />
  </a>
</td>
<td align="center">
  <a href="https://github.com/nathanlct">
    <img src="https://avatars.githubusercontent.com/u/33672752?v=4"  width="100px;" alt="Nathan Lichtlé" style="border-radius: 50%" />
  </a>
</td>
<td align="center">
  <a href="https://github.com/xiaomengy">
    <img src="https://avatars.githubusercontent.com/u/3357667?v=4" width="100px;" alt="Xiaomeng Yang" style="border-radius: 50%" />
  </a>
</td>
</tr>
<tr>
<td align="center">
  <a href="https://eugenevinitsky.github.io/">
    Eugene Vinitsky
  </a>
</td>
<td align="center">
  <a href="https://nathanlct.com/">
    Nathan Lichtlé
  </a>
</td>
<td align="center">
  <a href="https://github.com/xiaomengy">
    Xiaomeng Yang
  </a>
</td>
</tr>
</tbody>
</table>

# License

The majority of Nocturne is licensed under the MIT license, however portions of the project are available under separate license terms. The Waymo Motion Dataset License can be found at https://waymo.com/open/terms/.
