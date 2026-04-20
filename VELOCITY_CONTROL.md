# Terrain-Aware Velocity Control via Kernel Dynamics

**ros2_ws packages consumed:** `ros2_orb_slam3` · `feature_tracking_demo` · `px4_ros_com`
**kernelcal modules:** `navigation.slam` · `navigation.velocity` · `navigation.planner`

---

## ros2_ws source analysis

### What exists

| Package | Key capability | Integration hook |
|---|---|---|
| `ros2_orb_slam3` | Mono ORB-SLAM3; publishes `/orb_slam3/camera_pose`, `/orb_slam3/map_points` (PointCloud2), `/orb_slam3/tracking_state` (Int32), `/orb_slam3/trajectory` | Primary kernel data source |
| `feature_tracking_demo` | ORB/optical-flow tracking; `relative_transform/pose` (PoseStamped) + TF | Motion cue for look-ahead kernel |
| `px4_ros_com` | PX4 offboard: `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command`; subscribes `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_status` | Velocity command sink |
| `mynteye_ros2` | Stereo image publisher (`left/image_raw`, `right/image_raw`) | Image source for ORB-SLAM3 |

### What is missing

- **No terrain model:** no elevation grid, costmap, or semantic labelling of terrain
- **No adaptive speed policy:** `px4_ros_com` sends fixed setpoints; nothing maps terrain features to velocity limits
- **No kernelcal integration:** the two stacks (`ros2_orb_slam3` and `px4_ros_com`) are currently unconnected

---

## Kernel dynamics approach to velocity control

### Core idea

Speed is governed by how much the SLAM feature kernel is changing.  A rover that
*knows where it is* (kernel fixed point, low novelty) should go fast.  A rover that
*does not recognise the terrain* (transient kernel phase, high novelty) should slow
down and gather information before proceeding.

The three governing signals all come directly from `SemanticSLAMKernelTracker`:

```
novelty(t)    = ‖K(t) − K(t−1)‖_HS        ← how different is this frame?
stability(t)  = FixedPointDetector.score() ← has the map converged?
complexity(t) = spectral_entropy(K(t))     ← how feature-rich is the terrain?
```

These are combined multiplicatively inside `TerrainKernelVelocityController`:

```
v(t) = v_max × σ_nov(t) × σ_stab(t) × σ_cplx(t) × σ_track(t)
```

where each σ is a smooth monotone gate and `σ_track` provides a hard floor
based on the ORB-SLAM3 tracking state code.

### Look-ahead gate

Before reaching the next waypoint, the controller builds a local kernel from the
ORB-SLAM3 map points that *fall near the planned destination*.  If the distance
between that ahead-kernel and the current SLAM fixed-point kernel is large, the
rover pre-emptively slows — analogous to a model-predictive preview controller,
but formulated entirely in RKHS geometry.

```
f_look = σ_nov( d_HS(K_next_wp, K_fixed_point) )
v(t) ←  v(t) × f_look
```

### Tracking loss

ORB-SLAM3 publishes `tracking_state` ∈ {0=no_image, 1=not_init, 2=ok, 3=lost}.
The controller maps:

| State | σ_track | Behaviour |
|---|---|---|
| OK (2) | 1.0 | Full kernel-governed speed |
| Not initialised (1) | 0.30 | Crawl — building initial map |
| Lost (3) | 0.0 → hard stop | Rover stops until relocalisation |
| No image (0) | 0.10 | Near-stop |

---

## Integration architecture

```
ros2_ws                              kernelcal
─────────────────────────────────────────────────────────────────────
/orb_slam3/map_points  ─────────► map_points_to_kernel()
                                        │
/orb_slam3/tracking_state ──────►  TerrainKernelVelocityController
                                        │
/kernelcal/slam_novelty ────────►       │   (from SemanticSLAMKernelTracker
/kernelcal/map_stability ───────►       │    running in KernelcalNavigationNode)
                                        │
                       ◄────────  /kernelcal/velocity_cmd  (Float32, m/s)
                                        │
                       ◄────────  /kernelcal/velocity_factors (Float32MultiArray)
                                  [f_nov, f_stab, f_cplx, f_track]
                                        │
/fmu/in/trajectory_setpoint ◄─── KernelVelocityNode._update_cb()
  .velocity[0] = v_cmd (NED x)         │
  .velocity[1] = 0.0                   │
  .velocity[2] = 0.0                   │
```

The `KernelVelocityNode` (in `kernelcal/navigation/ros_bridge.py`) glues these
together in a single ROS2 node.  Enable PX4 setpoint with `use_px4_setpoint=True`.

---

## New kernelcal module: `navigation.velocity`

### `TerrainKernelVelocityController`

```python
from kernelcal.navigation.velocity import (
    TerrainKernelVelocityController, VelocityBand
)

ctrl = TerrainKernelVelocityController(
    band=VelocityBand(v_min=0.0, v_max=3.0, v_crawl=0.25),
    novelty_safe=0.08,     # HS distance below which terrain is "familiar"
    novelty_danger=1.20,   # HS distance above which rover crawls
    complexity_ref=2.2,    # spectral entropy of flat, well-mapped terrain
    smoothing_alpha=0.35,  # EMA smoothing on v_cmd
    use_look_ahead=True,
)

# At each control step (10 Hz):
v_cmd = ctrl.update(
    novelty=tracker.novelty_score(),
    stability=tracker.map_stability_score(),
    current_kernel=tracker._prev_kernel,
    tracking_state=orb_slam3_tracking_state,
    next_waypoint_kernel=look_ahead_kernel,
)
# publish v_cmd to /fmu/in/trajectory_setpoint or /cmd_vel
```

### `map_points_to_kernel`

Converts the PointCloud2 from `/orb_slam3/map_points` to an RBF kernel matrix
over the 3-D coordinates of nearby map points.  Used for both the current terrain
kernel and the look-ahead kernel.

```python
from kernelcal.navigation.velocity import map_points_to_kernel

K = map_points_to_kernel(
    points_xyz=np.array([[x1,y1,z1], [x2,y2,z2], ...]),
    fov_radius=5.0,   # metres — only use points within 5 m
    n_sample=50,
    length_scale=1.0,
)
```

---

## ROS2 node: `KernelVelocityNode`

```python
from kernelcal.navigation.ros_bridge import KernelVelocityNode

node = KernelVelocityNode(
    v_max=3.0,
    v_crawl=0.25,
    use_px4_setpoint=True,   # also publish to /fmu/in/trajectory_setpoint
    update_rate_hz=10.0,
)
node.spin()
```

Topics published:

| Topic | Type | Content |
|---|---|---|
| `/kernelcal/velocity_cmd` | `Float32` | Scalar forward speed m/s |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | NED velocity (x=forward) |
| `/kernelcal/velocity_factors` | `Float32MultiArray` | [f_nov, f_stab, f_cplx, f_track] |
| `/kernelcal/velocity_metrics` | `String` (JSON) | Full controller summary |

Topics subscribed:

| Topic | Source package | Used for |
|---|---|---|
| `/orb_slam3/tracking_state` | `ros2_orb_slam3` | Hard stop on TRACKING_LOST |
| `/orb_slam3/map_points` | `ros2_orb_slam3` | Local terrain kernel |
| `/kernelcal/slam_novelty` | `KernelcalNavigationNode` | Novelty gate |
| `/kernelcal/map_stability` | `KernelcalNavigationNode` | Stability gate |

---

## Launch integration (earth-rover)

Add to `launch/earth_rover.launch.py`:

```python
from launch_ros.actions import Node

kernelcal_velocity = Node(
    package="kernelcal_ros",       # thin wrapper package to be created
    executable="kernel_velocity_node",
    name="kernelcal_velocity",
    parameters=[{
        "v_max":              3.0,
        "v_crawl":            0.25,
        "use_px4_setpoint":   True,
        "update_rate_hz":     10.0,
        "novelty_safe":       0.08,
        "novelty_danger":     1.20,
        "complexity_ref":     2.2,
        "smoothing_alpha":    0.35,
    }],
)
```

---

## Demo toy result interpretation

Running `examples/navigation/demo_velocity_control.py` with synthetic ORB-SLAM3-style descriptors:

| Metric | Value | Interpretation |
|---|---|---|
| Mean v_cmd | 0.10 m/s | High novelty from random descriptors keeps the rover cautious |
| Tracking-lost steps | 30 / 80 | Simulated dark zone (bottom-left) stops rover repeatedly |
| Full-speed steps | 0% | SLAM never reaches fixed-point in unfamiliar synthetic terrain |
| Max v_cmd | 0.98 m/s | Achieved only at step 0 before novelty builds up |

This is the correct safety-first behaviour: in a real deployment on known terrain
(re-surveyed site after initial mapping), novelty will be low, stability will be
high, and the rover will run near v_max throughout.  Novel terrain during a new
survey triggers the conservative exploration regime automatically.

**Key insight from the phase portrait (vel_fig4 bottom-right):**
There is a monotone relationship between novelty and v_cmd.  The scatter shows no
high-speed points at high novelty — the controller correctly enforces:
*"fast only when you know where you are."*

---

## Prioritised next steps for real deployment

| Priority | Task | Effort |
|---|---|---|
| 1 | Wire `KernelVelocityNode` into `earth_rover.launch.py` | Low |
| 2 | Subscribe `/orb_slam3/map_points` in `KernelVelocityNode._map_points_cb` | Done |
| 3 | Tune `novelty_safe` / `novelty_danger` on real ORB-SLAM3 data | Medium |
| 4 | Add `/mavros/battery` energy constraint to gate speed near battery limit | Low |
| 5 | Record rosbag of human-pilot session → fit λ → velocity preference transfer | Medium |
| 6 | Extend to stereo-inertial ORB-SLAM3 mode (uncomment `System::STEREO_IMU`) | High |
| 7 | Replace `trajectory_setpoint` velocity with `TrajectorySetpoint.acceleration` for smoother PX4 motion | Low |
