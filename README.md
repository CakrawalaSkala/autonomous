# Autonomous Drone Vision System

This project integrates three interconnected subsystems to enable autonomous drone operation with real-time visual target tracking and waypoint navigation.

## Architecture Overview

The system comprises three major components working in concert:

1. **Simulation Layer** - Gazebo-based physics simulation with drone and environment models
2. **Autonomous Control Layer** - PyMAVLink-based flight controller communicating with simulated drone
3. **Computer Vision Layer** - YOLO-based real-time object detection processing camera stream from simulation

Data flows between layers via UDP sockets and MAVLink protocol over network connections. The drone operates in closed-loop feedback with vision processing continuously adjusting flight vectors based on target detection.

## Part 1: Gazebo Simulation Environment

Location: `model-gz/`

The simulation environment models a complete drone with camera and surrounding terrain using Gazebo 7+ simulation framework.

### Models

The `model-gz/models/` directory contains SDF (Simulation Description Format) models of the drone and environment:

- `iris_with_downward_camera/` - DJI Iris quadcopter equipped with downward-facing camera
- `iris_with_gimbal/` - Iris variant with gimbal mount for camera stabilization
- `iris_with_ardupilot/` - Iris configured for ArduCopter firmware integration
- `grass/` - Textured grass ground plane
- `camera_downward/` - Camera sensor model mounted on drone
- `runway/` - Landing zone model

Each model is defined in SDF XML format specifying geometry, physics properties, sensors, and visual appearance.

### Worlds

The `model-gz/worlds/` directory contains Gazebo world files (`.sdf`):

- `plain_grass.sdf` - Minimal world with grass ground plane and basic lighting
- `pole.sdf` - Extended world including target poles used for autonomous missions
- `texture.png` - Textured image applied to grass and target markers

The world file assembles multiple models and configures physics simulation parameters, lighting, and environment properties.

### Running the Simulation

Start Gazebo with the pole world file:

```bash
cd model-gz/worlds
gz sim -v4 -r pole.sdf
```

The `-v4` flag enables verbose logging; `-r` starts simulation in running state. The world renders a grass field with two target poles that the drone must navigate around.

## Part 2: Autonomous Control via PyMAVLink

Location: `script/`

This layer implements autonomous mission planning, flight control, and real-time target tracking using the MAVSDK library (MAVLink abstraction).

### Core Controller: main.py

The `script/main.py` script implements the autonomous flight controller:

- **Mission planning**: Generates figure-8 and circular waypoint patterns (`generate_capsule_waypoints()`) to systematically cover a defined area
- **Target tracking**: Receives normalized YOLO detection coordinates via UDP and computes velocity commands to center target in camera frame
- **Flight mode switching**: Transitions between OFFBOARD (autonomous) and AUTO (mission) modes based on target detection state
- **Real-time telemetry**: Receives GPS, altitude, battery, and vehicle state via MAVLink over UDP

Key operational parameters:

```python
speed = 25.0              # mission cruise speed (m/s)
acceptance_radius = 10.0  # waypoint completion threshold (m)
alt_mission = 10.0        # mission altitude AGL (m)
area_width = 100.0        # survey area dimensions (m)
fov = 70.0                # camera field of view (deg)
overlap = 0.30            # waypoint overlap for imaging
```

### UDP Communication

The controller operates as UDP server listening on `127.0.0.1:9000`:

- **Receives**: Target coordinates in normalized image space (dx, dy in [-1, 1] range)
  - dx: -1.0 (left) to +1.0 (right)
  - dy: -1.0 (top) to +1.0 (bottom)
- **Sends**: MAVLink telemetry to ground station (QGroundControl) on separate UDP ports

### Running the Controller

```bash
cd script
python main.py
```

The controller waits for YOLO stream and mission initialization before commencing autonomous operation.

### Supporting Scripts

- `trigger.py` - Simulates YOLO detection stream for testing (sends synthetic target coordinates)
- `generate-object.py` - Generates SDF models for targets (creates figure-8 marker patterns for mission area)

## Part 3: Computer Vision with YOLO

Location: `rdk/`

This subsystem runs real-time object detection on drone camera feed using YOLOv8 neural network.

### Model Files

The `rdk/model/` directory contains trained neural network weights in multiple formats:

- `best.pt` - PyTorch checkpoint (training format, largest file)
- `best.onnx` - ONNX format (portable, opset 11, for CPU inference)
- `best_ir8.onnx` - ONNX with IR version 8 (RDK runtime compatibility)
- `best_cut.onnx`, `best_nosimplify.onnx` - Alternative ONNX variants

### Live Inference: live-inference.py

The `script/live-inference.py` script processes video streams and sends detections to the flight controller:

- **Input sources**: webcam, video file, RTSP stream, or local socket
- **Processing pipeline**:
  - Frame capture at native resolution
  - Inference at fixed 640x640 resolution
  - Non-maximum suppression (NMS) with configurable IoU threshold
  - Bounding box drawing and FPS tracking
- **Output**: Normalized target coordinates sent to flight controller via UDP

Configuration:

```python
MODEL_PATH = "best.pt"       # model checkpoint
CONF_THRESH = 0.40           # detection confidence minimum
IOU_THRESH = 0.45            # NMS overlap threshold
IMG_SIZE = 640               # inference resolution (must match training)
DEVICE = ""                  # auto-select (GPU if available)
```

Controls during inference:

- Q or ESC - quit
- P - pause/resume
- S - save screenshot

### Model Preparation Pipeline

#### Export: export_to_onnx.py

Converts PyTorch checkpoint to ONNX format for deployment:

```bash
cd rdk
python export_to_onnx.py
```

Produces `model/best.onnx` with configurable opset version and optimization settings.

#### Calibration: calib_maker.py

Prepares image dataset for inference optimization:

```bash
cd rdk
python calib_maker.py
```

- Reads validation images from `rdk_val_images/`
- Applies letterboxing to 640x640 resolution
- Converts BGR to RGB, normalizes to [0, 1], transposes to CHW format
- Saves as raw binary float32 files in `rdk_calibration/`

### Drone Camera Integration

The vision system processes frames from the Gazebo simulation camera mounted on the iris drone model. Camera topic output is captured via ROS bridge or direct Gazebo plugin interface and fed to the YOLO inference pipeline.

## System Integration and Data Flow

### Startup Sequence

1. **Start Gazebo simulation**:
   ```bash
   cd model-gz/worlds && gz sim -v4 -r pole.sdf
   ```
   Initializes simulated world and drone at origin.

2. **Start SITL flight controller**:
   ```bash
   sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console --add-param-file $VISION/quad_sitl.param --out=udp:127.0.0.1:14550
   ```
   Launches ArduCopter firmware in software-in-the-loop (SITL) mode connected to Gazebo via JSON protocol.

3. **Start ground station (QGroundControl)**:
   ```bash
   exec/QGroundControl-x86_64.AppImage
   ```
   Connects to SITL flight controller for mission planning and telemetry visualization.

4. **Start vision inference**:
   ```bash
   cd script && python live-inference.py --source <camera_topic>
   ```
   Processes drone camera stream and sends detections.

5. **Start autonomous controller**:
   ```bash
   cd script && python main.py
   ```
   Listens for vision updates and executes mission with dynamic target tracking.

### Tmuxinator Configuration

The `drone.yml` file (tmuxinator session configuration) automates multi-window startup:

```bash
tmuxinator start drone
```

Launches four tmux panes:
- SITL (ArduCopter simulation)
- Gazebo (physics simulation)
- QGroundControl (ground station)
- (Optional) Python inference or controller

### Communication Channels

- **MAVLink over UDP**: SITL controller ↔ autonomous controller (telemetry, commands)
- **Vision UDP**: YOLO inference → autonomous controller (target coordinates)
- **Gazebo Plugin**: simulation ↔ SITL (physics feedback, sensor data)
- **QGroundControl UDP**: SITL ↔ ground station (mission upload, telemetry)

## Mission Execution Logic

The autonomous controller implements the following state machine:

### Mission Mode (AUTO)

- Executes pre-planned waypoint sequence generated from mission parameters
- Follows capsule-based path geometry ensuring smooth turns
- Nominal behavior: systematic area coverage at constant altitude and speed

### Tracking Mode (OFFBOARD)

- Triggered when YOLO detects target (confidence > threshold, time since last cooldown expired)
- Computes velocity command proportional to target offset in normalized image space
- Command equation:
  ```
  forward_velocity = const_speed
  lateral_velocity = K_lat * dx  (negative = left, positive = right)
  vertical_velocity = K_vert * dy (negative = up, positive = down)
  ```
- Maintains centering until target lost or time threshold exceeded

### Return to Mission

- Exits tracking mode after timeout or detection loss
- Resumes final waypoint or continues mission sequence
- Cooldown period prevents immediate re-triggering on same target

## Mission Parameter Configuration

Location: `script/main.py`

Mission geometry parameters define autonomous survey pattern:

```python
pole1 = (-35.36381716, 149.16499336)  # GPS coordinate
pole2 = (-35.36311189, 149.16291509)  # GPS coordinate

clearance = 30            # distance from obstacles (m)
precision = 15            # waypoint density
speed = 25.0              # cruise speed (m/s)
alt_mission = 10.0        # altitude above ground level (m)
area_width = 100.0        # survey swath width (m)
area_length = 30.0        # survey pass length (m)

lap = [
  [2 * np.pi, clearance, 1],        # full circle, tight radius
  [np.pi, clearance - 25, -1]       # semicircle, medium radius
]
```

Multiple laps enable multi-pass survey patterns with variable clearance from obstacles.

## Parameter Configuration

Location: `mav.parm` and `ardupilot/mav.parm`

Contains 1400+ ArduCopter firmware parameters controlling aircraft behavior:

- **ATC_** parameters: Attitude controller gains and limits
- **ACRO_** parameters: Acrobatic mode response
- **ANGLE_MAX**: Maximum lean angle (millidegrees)
- **ARMING_** parameters: Pre-flight checks and safety thresholds
- **AHRS_** parameters: Attitude estimation (EKF) configuration

Parameters are loaded at SITL startup via `--add-param-file` flag.

## Terrain Data

Location: `terrain/`

Directory contains SRTM (Shuttle Radar Topography Mission) elevation data files:

```
S21E128.DAT, S21E169.DAT
S34E136.DAT, S34E161.DAT
S36E149.DAT
```

These `.DAT` files provide digital elevation model (DEM) data for geographic regions. ArduPilot uses this data for terrain-relative altitude calculations and collision avoidance in AUTO missions.

## Development Workflow

### Building ArduPilot (Optional, if modifying firmware)

ArduPilot is provided as pre-built SITL binary, but modifications require rebuild:

```bash
cd ardupilot
./waf configure --board px4
./waf build
```

See `ardupilot/BUILD.md` for detailed build instructions and board-specific targets.

### Testing Vision Pipeline

Isolated testing without full system:

1. Test YOLO inference on recorded video:
   ```bash
   cd script
   python live-inference.py --source /path/to/video.mp4
   ```

2. Test UDP communication with mock stream:
   ```bash
   cd script
   python trigger.py &
   python live-inference.py --source 0
   ```

3. Test controller without drone (requires MAVLink server mock)

### Debugging

Enable verbose logging:

```python
DEBUG = True  # in script/main.py
```

Outputs:
- Waypoint transitions
- Target detection events
- Velocity commands
- Telemetry updates
- Error conditions

## Key Files Quick Reference

| Path | Purpose |
|------|---------|
| `script/main.py` | Autonomous flight controller |
| `script/live-inference.py` | YOLO detection and UDP stream |
| `rdk/model/best.pt` | Trained YOLO checkpoint |
| `rdk/export_to_onnx.py` | Model format conversion |
| `model-gz/worlds/pole.sdf` | Gazebo world definition |
| `mav.parm` | Firmware parameter configuration |
| `drone.yml` | Tmuxinator session automation |
| `ardupilot/` | ArduPilot flight firmware (C++) |

## External Dependencies

- **Gazebo 7+** - Physics simulation
- **ArduPilot** - Flight control firmware (included)
- **MAVSDK** - MAVLink abstraction library
- **YOLOv8** - Object detection (via Ultralytics)
- **OpenCV** - Image processing
- **NumPy** - Numerical computation
- **QGroundControl** - Ground station GUI

Install Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install mavsdk ultralytics opencv-python numpy
```

## License

This project integrates ArduPilot (GPL v3) and applies the same open-source licensing. See `ardupilot/COPYING.txt`.