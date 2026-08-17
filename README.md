
# Vision-Based Robotic Manipulation

A vision-based robotic manipulation system developed in **Python and MuJoCo** that enables a simulated robotic arm to detect an object, estimate its position from camera observations, calculate the required robot motion, grasp the object, transport it, and release it at a target location.

## Project Overview

This project explores the integration of **computer vision, robot kinematics, motion control, and robotic grasping** in a simulated environment.

The system follows a perception-to-action pipeline:

**Camera → Object Detection → Pixel Coordinates → World Coordinates → Inverse Kinematics → Robot Motion → Grasp → Transport → Release**

The project is designed as a modular robotics system where individual components can be tested independently before being integrated into the complete manipulation pipeline.

---

## Key Features

- Vision-based object detection
- MuJoCo robotic arm simulation
- Workspace camera simulation
- Pixel-to-world coordinate transformation
- Inverse kinematics for robot positioning
- Joint position control
- Robotic gripper control
- Object grasping
- Pick-and-lift manipulation
- Object transport and release
- Modular Python implementation
- Simulation result generation and visualization

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   MuJoCo Simulation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Workspace Camera  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Object Detection   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Pixel → World       │
                    │ Coordinate Mapping  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Inverse Kinematics  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Joint Position      │
                    │ Control             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Grasp / Pick Object │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transport & Release │
                    └─────────────────────┘
```

---

## Manipulation Pipeline

### 1. Scene Simulation

The robotic arm, workspace, object, and camera are defined in **MuJoCo XML models**.

The simulation provides the physical environment required for testing robotic manipulation without requiring physical hardware.

### 2. Camera-Based Perception

A simulated workspace camera observes the manipulation area.

The camera image is rendered from the MuJoCo environment and used as the input for object localization.

### 3. Object Detection

The system identifies the target object in the rendered camera image and determines its image-space location.

The detected object position is represented using pixel coordinates:

```text
(u, v)
```

where:

* `u` = horizontal pixel coordinate
* `v` = vertical pixel coordinate

### 4. Pixel-to-World Transformation

The detected pixel position is converted into a corresponding position in the robot's world coordinate system.

This allows the robot controller to determine where the object is physically located within the simulated workspace.

### 5. Inverse Kinematics

The desired end-effector position is converted into suitable robot joint configurations using inverse kinematics.

The resulting joint positions are then passed to the robot motion controller.

### 6. Grasping

The robotic arm approaches the detected object and positions the gripper around it.

The gripper is then controlled to establish the grasp.

### 7. Pick and Transport

After grasping the object, the arm lifts it from the workspace and moves toward the designated release location.

### 8. Release

The gripper opens at the target location, releasing the object and completing the manipulation task.

---

## Project Structure

```text
vision-based-robotic-manipulation/
│
├── models/
│   ├── basic_world.xml
│   ├── robot_arm.xml
│   └── robot_arm_backup.xml
│
├── outputs/
│   ├── detected_cube.png
│   └── workspace_camera.png
│
├── scripts/
│   ├── capture_camera.py
│   ├── detect_object.py
│   ├── grasp_test.py
│   ├── gripper_position_test.py
│   ├── gripper_test.py
│   ├── ik_test.py
│   ├── list_bodies.py
│   ├── pick_and_lift.py
│   ├── pixel_to_world.py
│   ├── position_control.py
│   ├── robot_motion.py
│   ├── simulate_arm.py
│   ├── simulate_basic.py
│   └── vertical_gripper_test.py
│
├── .gitignore
├── README.md
└── screenshot.png
```

---

## Technologies Used

| Technology         | Purpose                                      |
| ------------------ | -------------------------------------------- |
| Python             | Robotics control and system implementation   |
| MuJoCo             | Physics-based robotic simulation             |
| NumPy              | Numerical computation                        |
| Computer Vision    | Object localization from camera observations |
| Inverse Kinematics | Robot end-effector positioning               |
| Git                | Version control                              |
| GitHub             | Project hosting and collaboration            |

---

## Important Modules

### `capture_camera.py`

Captures and renders observations from the simulated workspace camera.

### `detect_object.py`

Handles object localization from the camera observation.

### `pixel_to_world.py`

Converts image-space coordinates into world-space coordinates for robot manipulation.

### `ik_test.py`

Tests inverse kinematics calculations and robot positioning.

### `position_control.py`

Controls the robot's joint positions.

### `robot_motion.py`

Provides robot movement functionality.

### `grasp_test.py`

Tests the robotic grasping process.

### `pick_and_lift.py`

Implements the pick-and-lift manipulation sequence.

### `simulate_arm.py`

Runs the robotic arm simulation.

---

## Example Workflow

A typical manipulation sequence is:

```text
Initialize MuJoCo environment
        ↓
Initialize robotic arm
        ↓
Render workspace camera
        ↓
Detect target object
        ↓
Obtain pixel coordinates
        ↓
Convert pixels → world coordinates
        ↓
Calculate inverse kinematics
        ↓
Move robot toward object
        ↓
Position gripper
        ↓
Close gripper
        ↓
Lift object
        ↓
Move to target position
        ↓
Open gripper
        ↓
Release object
```

---

## Results

The project generates camera observations and manipulation results that can be inspected in the `outputs/` directory.

Example outputs include:

* Detected object visualization
* Workspace camera view
* Robot manipulation simulation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Bhoomika229/vision-based-robotic-manipulation.git
cd vision-based-robotic-manipulation
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install mujoco numpy
```

---

## Running the Project

After activating the virtual environment, individual modules can be executed from the `scripts` directory.

For example:

```bash
python scripts/simulate_basic.py
```

To test the robotic arm:

```bash
python scripts/simulate_arm.py
```

To test object detection:

```bash
python scripts/detect_object.py
```

To test pixel-to-world conversion:

```bash
python scripts/pixel_to_world.py
```

To test the manipulation pipeline:

```bash
python scripts/pick_and_lift.py
```

---

## Development Approach

The system was developed incrementally by validating individual robotics components before integrating them into the complete manipulation pipeline.

The development stages included:

1. MuJoCo environment setup
2. Robot model validation
3. Camera rendering
4. Object detection
5. Pixel coordinate extraction
6. Pixel-to-world transformation
7. Inverse kinematics testing
8. Joint position control
9. Gripper testing
10. Pick-and-lift implementation
11. Integrated robotic manipulation

This modular approach makes the system easier to debug, extend, and experiment with.

---

## Future Improvements

Potential extensions include:

* Deep-learning-based object detection
* Real-time camera input
* 6-DoF pose estimation
* Improved grasp planning
* Collision-aware motion planning
* Reinforcement learning for manipulation
* Visual servoing
* Real robotic arm deployment
* ROS 2 integration
* Multi-object manipulation
* Dynamic object tracking

---

## Applications

The concepts explored in this project can be applied to:

* Industrial robotic manipulation
* Automated pick-and-place systems
* Warehouse robotics
* Vision-guided assembly
* Intelligent manufacturing
* Service robotics
* Autonomous robotic systems

---

## Author

**Bhoomika S Murthy**

M.Sc. Artificial Intelligence
Brandenburg University of Technology (BTU) Cottbus–Senftenberg

---

## License

This project is intended for educational, research, and portfolio purposes.




