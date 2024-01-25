# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Construct a scenarios.json file from a waymos protobuf."""

from collections import defaultdict
import functools
import math
import json
from typing import Any, Dict, Iterator, Optional

import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

# TODO(ev) remove hardcoding
ERR_VAL = -10000.0

def get_features_description(
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
    include_sdc_paths: bool = False,
    num_paths: Optional[int] = 45,
    num_points_per_path: Optional[int] = 800,
    num_tls: Optional[int] = 16,
) -> dict[str, tf.io.FixedLenFeature]:
  """Returns a dictionary of all features to be extracted.

  Args:
    max_num_objects: Max number of objects.
    max_num_rg_points: Max number of sampled roadgraph points.
    include_sdc_paths: Whether to include roadgraph traversal paths for the SDC.
    num_paths: Optional number of SDC paths. Must be defined if
      `include_sdc_paths` is True.
    num_points_per_path: Optional number of points per SDC path. Must be defined
      if `include_sdc_paths` is True.
    num_tls: Maximum number of traffic lights.

  Returns:
    Dictionary of all features to be extracted.

  Raises:
    ValueError: If `include_sdc_paths` is True but either `num_paths` or
      `num_points_per_path` is None.
  """
  if include_sdc_paths and (num_paths is None or num_points_per_path is None):
    raise ValueError(
        'num_paths and num_points_per_path must be defined if SDC '
        'paths are included (include_sdc_paths).'
    )

  roadgraph_features = {
      'roadgraph_samples/dir': tf.io.FixedLenFeature(
          [max_num_rg_points, 3], tf.float32, default_value=None
      ),
      'roadgraph_samples/id': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/type': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/valid': tf.io.FixedLenFeature(
          [max_num_rg_points, 1], tf.int64, default_value=None
      ),
      'roadgraph_samples/xyz': tf.io.FixedLenFeature(
          [max_num_rg_points, 3], tf.float32, default_value=None
      ),
  }

  # Features of other agents.
  state_features = {
      'state/id': tf.io.FixedLenFeature(
          [max_num_objects], tf.float32, default_value=None
      ),
      'state/type': tf.io.FixedLenFeature(
          [max_num_objects], tf.float32, default_value=None
      ),
      'state/is_sdc': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
      'state/tracks_to_predict': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
      'state/objects_of_interest': tf.io.FixedLenFeature(
          [max_num_objects], tf.int64, default_value=None
      ),
  }
  num_timesteps = {'past': 10, 'current': 1, 'future': 80}
  for time in ['past', 'current', 'future']:
    steps_to_extract = num_timesteps[time]
    state_time_features = {
        'state/%s/bbox_yaw'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/height'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/length'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/timestamp_micros'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.int64, default_value=None
        ),
        'state/%s/valid'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.int64, default_value=None
        ),
        'state/%s/vel_yaw'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/speed'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/velocity_x'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/velocity_y'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/width'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/x'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/y'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
        'state/%s/z'
        % time: tf.io.FixedLenFeature(
            [max_num_objects, steps_to_extract], tf.float32, default_value=None
        ),
    }
    state_features.update(state_time_features)

  traffic_light_features = {
      'traffic_light_state/current/state': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/valid': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/id': tf.io.FixedLenFeature(
          [1, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/current/x': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/y': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/z': tf.io.FixedLenFeature(
          [1, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/current/timestamp_micros': tf.io.FixedLenFeature(
          [
              1,
          ],
          tf.int64,
          default_value=None,
      ),
      'traffic_light_state/past/state': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/valid': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/id': tf.io.FixedLenFeature(
          [10, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/past/x': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/y': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/z': tf.io.FixedLenFeature(
          [10, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/past/timestamp_micros': tf.io.FixedLenFeature(
          [
              10,
          ],
          tf.int64,
          default_value=None,
      ),
      'traffic_light_state/future/state': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/valid': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/id': tf.io.FixedLenFeature(
          [80, num_tls], tf.int64, default_value=None
      ),
      'traffic_light_state/future/x': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/y': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/z': tf.io.FixedLenFeature(
          [80, num_tls], tf.float32, default_value=None
      ),
      'traffic_light_state/future/timestamp_micros': tf.io.FixedLenFeature(
          [
              80,
          ],
          tf.int64,
          default_value=None,
      ),
  }
  features_description = {}
  features_description.update(roadgraph_features)
  if include_sdc_paths:
    features_description.update({
        'path_samples/xyz': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path, 3], tf.float32, default_value=None
        ),
        'path_samples/valid': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.int64, default_value=None
        ),
        'path_samples/id': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.int64, default_value=None
        ),
        'path_samples/arc_length': tf.io.FixedLenFeature(
            [num_paths, num_points_per_path], tf.float32, default_value=None
        ),
        'path_samples/on_route': tf.io.FixedLenFeature(
            [num_paths, 1], tf.int64, default_value=None
        ),
    })
  features_description.update(state_features)
  features_description.update(traffic_light_features)
  return features_description


def tf_examples_dataset(
    path: str,
    preprocess_fn,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 100,
    repeat: Optional[int] = None,
    batch_dims = (),
    num_shards: int = 1,
    deterministic: bool = True,
    drop_remainder: bool = True,
    tf_data_service_address: Optional[str] = None,
    batch_by_scenario: bool = True,
) -> tf.data.Dataset:
  """Returns a dataset of Open Motion dataset TFExamples.

  Each TFExample contains data for the trajectory of all objects, the roadgraph,
  and traffic light states. See https://waymo.com/open/data/motion/tfexample
  for the data format definition.

  Args:
    path: The path to the dataset.
    data_format: Data format of the dataset.
    preprocess_fn: Function for parsing and preprocessing individual examples.
    shuffle_seed: Seed for shuffling. If left default (None), will not shuffle
      the dataset.
    shuffle_buffer_size: The size of the shuffle buffer.
    repeat: Number of times to repeat the dataset. Default (None) will repeat
      infinitely.
    batch_dims: List of size of batch dimensions. Multiple batch dimension can
      be used to provide inputs for multiple devices. E.g.
      [jax.local_device_count(), batch_size_per_device].
    num_shards: Number of shards for parallel loading, no effect on data
      returned.
    deterministic: Whether to use deterministic parallel processing.
    drop_remainder: Arg for tf.data.Dataset.batch. Set True to drop remainder if
      the last batch does not contains enough examples.
    tf_data_service_address: Set to use tf data service.
    batch_by_scenario: If True, one example in a returned batch is the entire
      scenario containing all objects; if False, the dataset will treat
      individual object trajectories as a training example rather than an entire
      scenario.

  Returns:
    A tf.data.Dataset of Open Motion Dataset tf.Example elements.
  """


  files_to_load = [path]
#   if '@' in os.path.basename(path):
#     files_to_load = generate_sharded_filenames(path)
#   if shuffle_seed:
#     random.seed(shuffle_seed)
#     random.shuffle(files_to_load)
  dataset_fn = tf.data.TFRecordDataset
  import ipdb; ipdb.set_trace()
  files = tf.data.Dataset.from_tensor_slices(files_to_load)
  # Split files across multiple processes for distributed training/eval.
#   files = files.shard(jax.process_count(), jax.process_index())

  def _make_dataset(
      shard_index: int, num_shards: int, local_files: tf.data.Dataset
  ):
    # Outer parallelism.
    local_files = local_files.shard(num_shards, shard_index)
    ds = local_files.interleave(
        dataset_fn,
        num_parallel_calls=AUTOTUNE,
        cycle_length=AUTOTUNE,
        deterministic=deterministic,
    )

    ds = ds.repeat(repeat)
    ds = ds.map(
        preprocess_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
    )
    if not batch_by_scenario:
      ds = ds.unbatch()
    if batch_dims:
      for batch_size in reversed(batch_dims):
        ds = ds.batch(
            batch_size,
            drop_remainder=drop_remainder,
            num_parallel_calls=AUTOTUNE,
            deterministic=deterministic,
        )
    return ds

  make_dataset_fn = functools.partial(
      _make_dataset, num_shards=num_shards, local_files=files
  )
  indices = tf.data.Dataset.range(num_shards)
  dataset = indices.interleave(
      make_dataset_fn, num_parallel_calls=AUTOTUNE, deterministic=deterministic
  )

  if tf_data_service_address is not None:
    dataset = dataset.apply(
        tf.data.experimental.service.distribute(
            processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
            service=tf_data_service_address,
        )
    )
  return dataset.prefetch(AUTOTUNE)

def preprocess_serialized_womd_data(
    serialized: bytes
) -> dict[str, tf.Tensor]:
  """Parses serialized tf example into tf Tensor dict."""
  womd_features = get_features_description(
  )

  deserialized = tf.io.parse_example(serialized, womd_features)
  return preprocess_womd_example(
      deserialized,
      aggregate_timesteps=False,
      max_num_objects=128,
  )
  
def preprocess_womd_example(
    example: dict[str, tf.Tensor],
    aggregate_timesteps: bool,
    max_num_objects: Optional[int] = None,
) -> dict[str, tf.Tensor]:
  """Preprocesses dict of tf tensors, keyed by str."""

  processed = example

  if max_num_objects is not None:
    # TODO check sdc included if it is needed.
    return {
        k: v[:max_num_objects] if k.startswith('state/') else v
        for k, v in processed.items()
    }
  else:
    return processed

def waymo_to_scenario(scenario_path: str,
                      no_tl: bool = False) -> None:
    """Dump a JSON File containing the protobuf parsed into the right format.

    Args
    ----
        scenario_path (str): path to dump the json file
        protobuf (scenario_pb2.Scenario): the protobuf we are converting
        no_tl (bool, optional): If true, environments with traffic lights are not dumped.
    """
    dataset = tf_examples_dataset(
        path=scenario_path,
        preprocess_fn=preprocess_serialized_womd_data,
    )

    import ipdb; ipdb.set_trace()
    
    tl_dict = defaultdict(lambda: {
        'state': [],
        'x': [],
        'y': [],
        'time_index': []
    })
    all_keys = ['state', 'x', 'y']
    i = 0
    for dynamic_map_state in protobuf.dynamic_map_states:
        traffic_light_dict = _init_tl_object(dynamic_map_state)
        # there is a traffic light but we don't want traffic light scenes so just return
        if (no_tl and len(traffic_light_dict) > 0):
            return
        for id, value in traffic_light_dict.items():
            for state_key in all_keys:
                tl_dict[id][state_key].append(value[state_key])
            tl_dict[id]['time_index'].append(i)
        i += 1

    # Construct the object states
    objects = []
    for track in protobuf.tracks:
        obj = _init_object(track)
        if obj is not None:
            objects.append(obj)

    # Construct the map states
    roads = []
    for map_feature in protobuf.map_features:
        road = _init_road(map_feature)
        if road is not None:
            roads.append(road)

    scenario = {
        "name": scenario_path.split('/')[-1],
        "objects": objects,
        "roads": roads,
        "tl_states": tl_dict
    }
    with open(scenario_path, "w") as f:
        json.dump(scenario, f)