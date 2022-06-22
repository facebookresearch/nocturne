# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Storage for SLURM running utilities."""


class Overrides(object):
    """Utility class used to convert commands into a bash runnable string."""

    def __init__(self):
        """Initialize class."""
        self.kvs = dict()

    def add(self, key, values):
        """Add each of the desired key value pairs into a dict."""
        value = ','.join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        """Append the keys together into a command that can be run."""
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd
