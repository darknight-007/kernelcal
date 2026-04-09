"""
bloom_maxcal_sim — ROS2 package for MaxCal-based algae bloom following.

Modules
-------
bloom_field          : Spatiotemporally varying double-gyre advecting Gaussian bloom
rover_model          : 2D differential-drive rover kinematics and sensing
maxcal_bloom_follower: MaxCal-based bloom-following velocity controller
nodes/               : ROS2 node wrappers for each component
"""

from .bloom_field import AlgaeBloomField, BloomFieldConfig, DoubleGyreParams, BloomPatch
from .rover_model import DifferentialDriveRover, RoverConfig, WaypointVelocityController
from .maxcal_bloom_follower import MaxCalBloomFollower, MaxCalConfig

__all__ = [
    'AlgaeBloomField', 'BloomFieldConfig', 'DoubleGyreParams', 'BloomPatch',
    'DifferentialDriveRover', 'RoverConfig', 'WaypointVelocityController',
    'MaxCalBloomFollower', 'MaxCalConfig',
]
