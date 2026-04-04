# kernelcal × Earth Rover: Autonomous Navigation Integration

**Codebase:** [`Earth-Innovation-Hub/earth-rover`](https://github.com/Earth-Innovation-Hub/earth-rover)
**ROS2 package:** `deepgis_vehicles`
**Key stack:** MAVROS/PX4 · ORB-SLAM3 · ROS2 · DeepGIS telemetry

---

## Core idea

The Earth Rover's navigation stack is currently open-loop in the representational sense:
the vehicle follows GPS waypoints or a learned path, but the *kernel* — the feature
structure that determines which differences in the environment are actionable — is fixed
at design time and never updated in the field.

Kernel dynamics closes this loop. The human pilot, the SLAM system, the semantic detector, and
the energy budget all produce signals that should evolve the navigation kernel over time.
MaxCal governs *how* it evolves: the path through kernel space that maximises entropy
subject to the constraints actually observed in the field.

---

## Thread 1 — Semantic SLAM as kernel trajectory

### Current state
ORB-SLAM3 extracts ORB descriptors (binary feature vectors) at keypoints. Loop closure
fires when the bag-of-words score between the current frame and a stored keyframe
exceeds a threshold. This is a fixed kernel over image space — the ORB descriptor kernel
does not adapt to the environment.

### kernelcal connection
ORB descriptors define a kernel matrix over matched keypoints:

    K_SLAM[i,j] = k(d_i, d_j)    (e.g. RBF over descriptor distance)

A `KernelTrajectory` over successive keyframes tracks how the feature structure of the
map evolves as the rover explores. Two uses:

**Loop closure confidence via HS distance**
Rather than a binary threshold, loop closure confidence becomes a continuous metric:

    confidence(t) = 1 / (1 + d_HS(K_current, K_stored))

Small HS distance → high confidence the rover is in a known location.
Large HS distance → novel environment, no reliable loop closure.

**Novelty-driven exploration**
The segment distance ‖K(t) - K(t-1)‖_HS measures how much the representation changed
between consecutive frames. High values flag genuinely novel terrain, triggering
prioritised observation rather than blind forward progress.

**Fixed-point = well-mapped region**
When ‖K(t) - K(t-1)‖_HS falls below threshold, the rover has reached a kernel fixed
point: the SLAM map is self-consistent and loop closure is reliable. This is the
formal condition for "the rover knows where it is."

### Code
```python
from kernelcal.navigation.slam import SemanticSLAMKernelTracker

tracker = SemanticSLAMKernelTracker(descriptor_dim=32)

# Call on each SLAM keyframe
tracker.update(descriptors=orb_descriptors, keyframe_id=kf_id)

print(tracker.novelty_score())          # HS distance from previous frame
print(tracker.loop_closure_confidence(stored_keyframe_descriptors))
print(tracker.is_well_mapped())         # fixed-point condition
```

---

## Thread 2 — Informative path planning via MaxCal

### Current state
The rover executes GPS waypoint missions planned in QGroundControl, or replays a
learned human-pilot path. There is no onboard mechanism for selecting *which* locations to
visit based on expected information gain.

### kernelcal connection
The `MaxCalSampler` replaces the static waypoint list with an entropy-maximising
distribution over candidate locations, subject to:

- **Energy constraint** `⟨E_i⟩_p ≤ E_budget`: expected energy per waypoint ≤ remaining
  battery budget (from `/mavros/battery`)
- **Semantic reward constraint** `⟨I_i⟩_p ≥ I_target`: expected information gain from
  semantic detections (SLAM novelty score, object detection confidence)
- **Coverage constraint**: expected visit count per region ≥ minimum threshold

The Lagrange multipliers are fitted in real-time as the battery drains and the semantic
map fills in — the path distribution automatically shifts to high-value, energy-efficient
regions.

**Fixed point = optimal patrol**
When the sampler's kernel stabilises (FixedPointDetector fires), the rover has found its
self-consistent patrol: the loop that maximises representational gain per joule given the
current environment. In stable ecosystems this is a fixed, efficient circuit. After an
ecological disturbance (fallen tree, erosion event), the kernel destabilises → transient
phase → new fixed point encoding the updated environment.

### Code
```python
from kernelcal.navigation.planner import InformativePathPlanner

planner = InformativePathPlanner(
    candidate_waypoints=grid_of_locations,   # (N, 2) lon/lat
    energy_budget_joules=battery_remaining,
    semantic_reward_fn=slam_novelty_scores,
)

planner.update(
    battery_joules_remaining=mavros_battery.remaining,
    semantic_scores=current_novelty_map,
)

next_waypoint = planner.next_waypoint()
print(planner.is_at_fixed_point())   # True = stable optimal patrol found
```

---

## Thread 3 — Learning from the human pilot (inverse MaxCal)

### Current state
The rover can record and replay a human pilot's path. This is pure imitation — no
generalisation, no understanding of *why* the pilot chose that path.

### kernelcal connection
The pilot's demonstrated trajectory is a sample from the pilot's implicit path
distribution. Inverse MaxCal recovers the Lagrange multipliers that make the
MaxCal distribution most consistent with the demonstrated path:

    p_pilot[γ] ∝ q[γ] exp(−λ · f(γ))

where `f(γ)` are observable features of the path (energy used, semantic novelty
encountered, obstacles avoided, speed profile) and `λ` are learned from the
demonstration.

**What this recovers:**
- `λ_energy` — how much the pilot penalises energy expenditure per unit of
  information gain: their implicit *thermodynamic efficiency preference*
- `λ_obstacle` — implicit obstacle avoidance cost kernel
- `λ_novelty` — how strongly the pilot seeks novel features vs. familiar paths

Once `λ` is learned, the rover can **generalise** to new environments: it generates
the MaxCal distribution with the learned multipliers applied to the new terrain's
feature matrix, producing human-pilot-consistent paths in places the pilot has never been.

**Kernel fixed point as the pilot's "home range"**
The pilot's kernel fixed point is the region of the environment where their path
distribution has stabilised — their preferred patrol. Deviations from the fixed point
(the pilot exploring somewhere new) register as transient phases that update `λ`.

### Code
```python
from kernelcal.navigation.pilot import HumanPilotDemonstrationLearner

learner = HumanPilotDemonstrationLearner(
    feature_fns=[energy_feature, novelty_feature, obstacle_proximity_feature],
)

# Feed recorded human-pilot paths (list of waypoint sequences)
for recorded_path in rosbag_paths:
    learner.add_demonstration(recorded_path)

learner.fit()   # inverse MaxCal: recover Lagrange multipliers

# Transfer to new environment
new_planner = learner.make_planner(candidate_waypoints=new_grid)
next_wp = new_planner.next_waypoint()   # human-pilot-consistent path in novel terrain
```

---

## Thread 4 — Obstacle avoidance as kernel exclusion

### Current state
Obstacle avoidance is handled by the PX4 autopilot or manually. There is no semantic
representation of *why* a region is an obstacle or how permanent it is.

### kernelcal connection
Obstacles are regions where the navigation kernel has zero or near-zero weight.
Two obstacle types map to different kernel regimes:

**Static obstacles** (rocks, walls, curbs) → kernel fixed points with zero probability.
The FixedPointDetector identifies these as stable features of the map.

**Dynamic obstacles** (pedestrians, cyclists, parked vehicles) → transient kernel
perturbations. The HS distance between the current kernel and the stable-map kernel
spikes when a dynamic obstacle enters the field. The sampler's distribution
automatically excludes the region while the distance is elevated, then recovers
when the obstacle leaves — no explicit tracking required.

**Semantic obstacle kernels**
Each obstacle category (rock, human, vehicle) has a characteristic RKHS norm from its
detector embedding. The assembly complexity map scores regions by their embedding norm
— high-complexity regions with unfamiliar kernel structure are treated as high-uncertainty
obstacles by the planner, triggering conservative speed reduction.

---

## Thread 5 — Energy-aware thermodynamic navigation

### Current state
The rover's energy system (LiFePO4 traction battery 25Ah/57.6V, LiFePO4 avionics
battery 100Ah/14.4V, 200W solar MPPT, regenerative braking) is not connected to
navigation decisions. Battery state is available via `/mavros/battery` but is not
currently subscribed to by any planning node.

### kernelcal connection
The thermodynamic bound `δW ≥ k_BT δI_k` applies directly to rover navigation:

- **δW** = energy consumed travelling to a waypoint (measurable from battery drain)
- **δI_k** = semantic information gained at that waypoint (SLAM novelty + object detections)
- **k_BT** = a calibrated constant that sets the minimum energy cost per nat of new
  environmental information

Tracking this ratio over a mission gives a *thermodynamic efficiency curve* for the
rover's exploration behaviour. An efficient mission follows the theoretical minimum:
each joule spent returns the maximum possible representational gain. Inefficient
stretches (traversing already-mapped, featureless terrain) show up as low efficiency
and trigger the planner to re-route.

**Solar income as information budget**
On a sunny day, the 200W solar panel provides ~1 kJ every 5 seconds. The Landauer
bound says each nat of new environmental information costs at minimum:

    δW_min = k_B × T_room × 1 nat ≈ 4 × 10^{-21} J

The solar panel thus provides an enormous information budget — the bottleneck is
mobility energy (drive motor), not thermodynamic cost. This means the rover should
maximise information gain per metre driven, not per joule of computation.

**ROS2 integration**
```python
from kernelcal.navigation.ros_bridge import ROSPowerMonitor, InformativeWaypointPublisher

# Subscribes to /mavros/battery instead of nvidia-smi
power_monitor = ROSPowerMonitor(node, battery_topic='/mavros/battery')

# Publishes MaxCal-optimal next waypoint to /kernelcal/next_waypoint
wp_publisher = InformativeWaypointPublisher(node, planner=planner)
```

---

## Architecture: closing the loop

```
              ┌─────────────────────────────────────────┐
              │            Earth Rover                  │
              │                                         │
  Camera ─────┤─► ORB-SLAM3                             │
              │       │                                 │
              │       ▼                                 │
              │  SemanticSLAMKernelTracker               │
              │   novelty_score()   ◄──────────────┐    │
              │   loop_closure_confidence()         │    │
              │       │                            │    │
              │       ▼                            │    │
              │  InformativePathPlanner             │    │
  /mavros/    │   (MaxCalSampler + energy           │    │
  battery ────┤─►  + semantic constraints)          │    │
              │       │                            │    │
              │       ▼                            │    │
              │  /kernelcal/next_waypoint  ─► PX4 ─┘    │
              │                                         │
  Pilot rosbag┤─► HumanPilotDemonstrationLearner         │
              │        ▼                                │
              │   Learned λ (generalised to new terrain)│
              │                                         │
  DeepGIS ◄───┤─ telemetry + kernelcal metrics          │
  telemetry   │  (entropy, novelty, efficiency)         │
              └─────────────────────────────────────────┘
```

---

## Prioritised implementation roadmap

| Priority | Component | Effort | Unlocks |
|---|---|---|---|
| 1 | `ROSPowerMonitor` — subscribe `/mavros/battery` | Low | Landauer bound on real hardware |
| 2 | `SemanticSLAMKernelTracker` — HS distance from ORB-SLAM3 descriptors | Medium | Continuous loop-closure confidence, novelty map |
| 3 | `InformativePathPlanner` — MaxCalSampler with energy + novelty constraints | Medium | Replaces static waypoint missions |
| 4 | Telemetry extension — add kernelcal metrics to `deepgis_telemetry_publisher.py` | Low | DeepGIS visibility of kernel dynamics |
| 5 | `HumanPilotDemonstrationLearner` — inverse MaxCal from rosbag | High | Human-pilot-consistent generalisation to new terrain |
| 6 | Obstacle kernel exclusion — HS spike detection for dynamic obstacles | Medium | Reactive avoidance without explicit tracking |

---

## Related files in earth-rover

| File | Role | kernelcal integration point |
|---|---|---|
| `scripts/deepgis_telemetry_publisher.py` | HTTP telemetry to deepgis.org | Add `entropy_nats`, `novelty_score`, `efficiency` fields |
| `src/vehicle_interface_node.cpp` | MAVROS bridge | Subscribe `/mavros/battery` → `ROSPowerMonitor` |
| `launch/earth_rover.launch.py` | Full system launch | Add `kernelcal_navigation` node |
| `config/deepgis_telemetry.yaml` | Telemetry params | Add `kernelcal_publish_rate`, `novelty_threshold` |
| `launch/full_system.launch.py` | All nodes | Integrate kernelcal after SLAM init |
