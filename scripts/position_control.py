from pathlib import Path
import time
import numpy as np

import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# 1. Locate the robot model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# ---------------------------------------------------------
# 2. Load MuJoCo model and simulation state
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


# ---------------------------------------------------------
# 3. Controller parameters
# ---------------------------------------------------------

# Proportional gain
Kp = 20.0

# Derivative gain
Kd = 2.0


# Desired joint positions in radians
target = np.array([
    0.5,    # Joint 1
    0.7,    # Joint 2
    -0.5    # Joint 3
])


print("Position controller loaded.")
print(f"Target joint positions: {target}")


# ---------------------------------------------------------
# 4. Open MuJoCo viewer
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(
    model,
    data,
    show_left_ui=False,
    show_right_ui=False
) as viewer:

    print("Position control started.")

    while viewer.is_running():

        # -------------------------------------------------
        # Read current joint positions
        # -------------------------------------------------

        q = data.qpos[:3]

        # Current joint velocities
        qd = data.qvel[:3]


        # -------------------------------------------------
        # Calculate position error
        # -------------------------------------------------

        position_error = target - q


        # -------------------------------------------------
        # PD controller
        # -------------------------------------------------

        torque = Kp * position_error - Kd * qd


        # -------------------------------------------------
        # Send torque commands to motors
        # -------------------------------------------------

        data.ctrl[:] = torque


        # -------------------------------------------------
        # Advance physics
        # -------------------------------------------------

        mujoco.mj_step(model, data)


        # -------------------------------------------------
        # Display controller state
        # -------------------------------------------------

        print(
            f"\rTarget: "
            f"{target[0]: .2f}, "
            f"{target[1]: .2f}, "
            f"{target[2]: .2f}   "
            f"| Current: "
            f"{q[0]: .2f}, "
            f"{q[1]: .2f}, "
            f"{q[2]: .2f}",
            end=""
        )


        # Update viewer
        viewer.sync()

        time.sleep(0.01)