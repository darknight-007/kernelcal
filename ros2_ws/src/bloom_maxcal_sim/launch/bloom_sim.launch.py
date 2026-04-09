"""
Launch file: bloom_maxcal_sim full simulation.

Starts four nodes in a coordinated simulation:
  1. bloom_field_node    — generates and publishes the advecting Gaussian bloom
  2. rover_sim_node      — simulates the 2D differential-drive rover
  3. maxcal_controller_node — MaxCal bloom-following velocity controller
  4. visualizer_node     — live matplotlib display

Usage
-----
    ros2 launch bloom_maxcal_sim bloom_sim.launch.py

With custom parameters:
    ros2 launch bloom_maxcal_sim bloom_sim.launch.py \
        gyre_A:=0.15 gyre_eps:=0.30 n_candidates:=48 v_max:=1.5

Arguments (all optional, with defaults)
----------------------------------------
domain_lx       float  100.0   Domain width (m)
domain_ly       float  100.0   Domain height (m)
grid_nx         int    120     Bloom grid columns
grid_ny         int    120     Bloom grid rows
gyre_A          float  0.10    Double-gyre stream amplitude (m²/s)
gyre_eps        float  0.25    Gyre oscillation amplitude ε
bloom_sim_dt    float  0.5     Bloom simulation step (s)
bloom_rate      float  2.0     Bloom publish frequency (Hz)
sigma_wind      float  0.008   Turbulent patch noise
rover_x0        float  50.0    Rover initial x (m)
rover_y0        float  50.0    Rover initial y (m)
rover_dt        float  0.05    Rover integration step (s)
rover_rate      float  20.0    Rover publish frequency (Hz)
v_max           float  1.2     Rover max speed (m/s)
control_rate    float  2.0     MaxCal controller frequency (Hz)
n_candidates    int    32      MaxCal candidate positions per step
lookahead_min   float  3.0     Inner lookahead ring (m)
lookahead_max   float  10.0    Outer lookahead ring (m)
visualize       bool   true    Launch the matplotlib visualizer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Argument declarations
# ---------------------------------------------------------------------------

_ARGS = [
    ('domain_lx',      '100.0',  'Domain width (m)'),
    ('domain_ly',      '100.0',  'Domain height (m)'),
    ('grid_nx',        '120',    'Bloom grid columns'),
    ('grid_ny',        '120',    'Bloom grid rows'),
    ('gyre_A',         '0.10',   'Double-gyre stream amplitude'),
    ('gyre_eps',       '0.25',   'Gyre oscillation amplitude'),
    ('gyre_omega',     '0.02094', 'Gyre oscillation angular frequency (rad/s)'),
    ('bloom_sim_dt',   '0.5',    'Bloom simulation step (s)'),
    ('bloom_rate',     '2.0',    'Bloom publish frequency (Hz)'),
    ('sigma_wind',     '0.008',  'Turbulent patch noise (m/sqrt(s))'),
    ('rover_x0',       '50.0',   'Rover initial x (m)'),
    ('rover_y0',       '50.0',   'Rover initial y (m)'),
    ('rover_theta0',   '0.0',    'Rover initial heading (rad)'),
    ('rover_dt',       '0.05',   'Rover integration step (s)'),
    ('rover_rate',     '20.0',   'Rover publish frequency (Hz)'),
    ('v_max',          '1.2',    'Rover max linear speed (m/s)'),
    ('omega_max',      '1.2',    'Rover max angular speed (rad/s)'),
    ('control_rate',   '2.0',    'MaxCal control rate (Hz)'),
    ('n_candidates',   '32',     'Candidate positions per MaxCal step'),
    ('lookahead_min',  '3.0',    'Inner lookahead ring radius (m)'),
    ('lookahead_max',  '10.0',   'Outer lookahead ring radius (m)'),
    ('sigma_q',        '6.0',    'Reference prior width (m)'),
    ('bloom_target_q', '0.65',   'Bloom concentration target quantile'),
    ('grad_target_q',  '0.55',   'Gradient target quantile'),
    ('kernel_ls',      '5.0',    'RBF kernel length scale (m)'),
    ('visualize',      'true',   'Launch matplotlib visualizer'),
    ('vis_rate',       '2.0',    'Visualizer refresh rate (Hz)'),
    ('seed_bloom',     '42',     'Bloom RNG seed'),
    ('seed_rover',     '123',    'Rover RNG seed'),
    ('seed_ctrl',      '7',      'Controller RNG seed'),
]


def generate_launch_description() -> LaunchDescription:
    declared = [
        DeclareLaunchArgument(name, default_value=default, description=desc)
        for name, default, desc in _ARGS
    ]

    lc = {name: LaunchConfiguration(name) for name, _, _ in _ARGS}

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    bloom_node = Node(
        package='bloom_maxcal_sim',
        executable='bloom_field_node',
        name='bloom_field_node',
        output='screen',
        parameters=[{
            'domain_lx':    lc['domain_lx'],
            'domain_ly':    lc['domain_ly'],
            'grid_nx':      lc['grid_nx'],
            'grid_ny':      lc['grid_ny'],
            'gyre_A':       lc['gyre_A'],
            'gyre_eps':     lc['gyre_eps'],
            'gyre_omega':   lc['gyre_omega'],
            'sim_dt':       lc['bloom_sim_dt'],
            'publish_rate': lc['bloom_rate'],
            'sigma_wind':   lc['sigma_wind'],
            'seed':         lc['seed_bloom'],
        }],
    )

    rover_node = Node(
        package='bloom_maxcal_sim',
        executable='rover_sim_node',
        name='rover_sim_node',
        output='screen',
        parameters=[{
            'x0':           lc['rover_x0'],
            'y0':           lc['rover_y0'],
            'theta0':       lc['rover_theta0'],
            'sim_dt':       lc['rover_dt'],
            'publish_rate': lc['rover_rate'],
            'v_max':        lc['v_max'],
            'omega_max':    lc['omega_max'],
            'domain_lx':    lc['domain_lx'],
            'domain_ly':    lc['domain_ly'],
            'seed':         lc['seed_rover'],
        }],
    )

    controller_node = Node(
        package='bloom_maxcal_sim',
        executable='maxcal_controller_node',
        name='maxcal_controller_node',
        output='screen',
        parameters=[{
            'control_rate':       lc['control_rate'],
            'n_candidates':       lc['n_candidates'],
            'lookahead_min':      lc['lookahead_min'],
            'lookahead_max':      lc['lookahead_max'],
            'sigma_q':            lc['sigma_q'],
            'bloom_target_q':     lc['bloom_target_q'],
            'gradient_target_q':  lc['grad_target_q'],
            'kernel_length_scale': lc['kernel_ls'],
            'v_max':              lc['v_max'],
            'domain_lx':          lc['domain_lx'],
            'domain_ly':          lc['domain_ly'],
            'seed':               lc['seed_ctrl'],
        }],
    )

    visualizer_node = Node(
        package='bloom_maxcal_sim',
        executable='visualizer_node',
        name='visualizer_node',
        output='screen',
        condition=IfCondition(lc['visualize']),
        parameters=[{
            'update_rate': lc['vis_rate'],
        }],
    )

    return LaunchDescription(
        declared + [bloom_node, rover_node, controller_node, visualizer_node]
    )
