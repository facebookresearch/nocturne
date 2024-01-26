# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Construct a scenarios.json file from a waymos protobuf."""

from collections import defaultdict
import math
import json
from typing import Any, Dict, Iterator, Optional

import tensorflow as tf
from waymo_open_dataset.protos import map_pb2, scenario_pb2

# TODO(ev) remove hardcoding
ERR_VAL = -10000.0

_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}

_WAYMO_ROAD_STR = {
    map_pb2.TrafficSignalLaneState.LANE_STATE_UNKNOWN: "unknown",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: "arrow_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: "arrow_caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: "arrow_go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: "stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: "caution",
    map_pb2.TrafficSignalLaneState.LANE_STATE_GO: "go",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: "flashing_stop",
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION:
    "flashing_caution",
}


def _parse_object_state(
        states: scenario_pb2.ObjectState,
        final_state: scenario_pb2.ObjectState) -> Dict[str, Any]:
    """Construct a dict representing the trajectory and goals of an object.

    Args:
        states (scenario_pb2.ObjectState): Protobuf of object state
        final_state (scenario_pb2.ObjectState): Protobuf of last valid object state.

    Returns
    -------
        Dict[str, Any]: Dict representing an object.
    """
    return {
        "position": [{
            "x": state.center_x,
            "y": state.center_y
        } if state.valid else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "width":
        final_state.width,
        "length":
        final_state.length,
        "heading": [
            math.degrees(state.heading) if state.valid else ERR_VAL
            for state in states
        ],  # Use rad here?
        "velocity": [{
            "x": state.velocity_x,
            "y": state.velocity_y
        } if state.valid else {
            "x": ERR_VAL,
            "y": ERR_VAL
        } for state in states],
        "valid": [state.valid for state in states],
        "goalPosition": {
            "x": final_state.center_x,
            "y": final_state.center_y
        }
    }


def _init_tl_object(track):
    """Construct a dict representing the traffic light states."""
    returned_dict = {}
    for lane_state in track.lane_states:
        returned_dict[lane_state.lane] = {
            'state': _WAYMO_ROAD_STR[lane_state.state],
            'x': lane_state.stop_point.x,
            'y': lane_state.stop_point.y
        }
    return returned_dict


def _init_object(track: scenario_pb2.Track) -> Optional[Dict[str, Any]]:
    """Construct a dict representing the state of the object (vehicle, cyclist, pedestrian).

    Args:
        track (scenario_pb2.Track): protobuf representing the scenario

    Returns
    -------
        Optional[Dict[str, Any]]: dict representing the trajectory and velocity of an object.
    """
    final_valid_index = 0
    for i, state in enumerate(track.states):
        if state.valid:
            final_valid_index = i

    obj = _parse_object_state(track.states, track.states[final_valid_index])
    obj["type"] = _WAYMO_OBJECT_STR[track.object_type]
    return obj


def _init_road(map_feature: map_pb2.MapFeature) -> Optional[Dict[str, Any]]:
    """Convert an element of the map protobuf to a dict representing its coordinates and type."""
    feature = map_feature.WhichOneof("feature_data")
    if feature == 'stop_sign':
        p = getattr(map_feature,
                    map_feature.WhichOneof("feature_data")).position
        geometry = [{"x": p.x, "y": p.y}]
    elif feature != 'crosswalk' and feature != 'speed_bump' and feature != 'driveway':
        geometry = [{
            "x": p.x,
            "y": p.y
        } for p in getattr(map_feature, map_feature.WhichOneof(
            "feature_data")).polyline]
    else:
        geometry = [{
            "x": p.x,
            "y": p.y
        } for p in getattr(map_feature, map_feature.WhichOneof(
            "feature_data")).polygon]
    return {
        "geometry": geometry,
        "type": map_feature.WhichOneof("feature_data"),
    }


def load_protobuf(protobuf_path: str) -> Iterator[scenario_pb2.Scenario]:
    """Yield the sharded protobufs from the TFRecord."""
    dataset = tf.data.TFRecordDataset(protobuf_path, compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))
        yield scenario


def waymo_to_scenario(scenario_path: str,
                      protobuf: scenario_pb2.Scenario,
                      no_tl: bool = False) -> None:
    """Dump a JSON File containing the protobuf parsed into the right format.

    Args
    ----
        scenario_path (str): path to dump the json file
        protobuf (scenario_pb2.Scenario): the protobuf we are converting
        no_tl (bool, optional): If true, environments with traffic lights are not dumped.
    """
    # read the protobuf file to get the right state

    # write the json file
    # construct the road geometries
    # place the initial position of the vehicles

    # Construct the traffic light states
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
    for i, track in enumerate(protobuf.tracks):
        obj = _init_object(track)
        if i == protobuf.sdc_track_index:
            obj['is_av'] = 1
        else:
            obj['is_av'] = 0
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