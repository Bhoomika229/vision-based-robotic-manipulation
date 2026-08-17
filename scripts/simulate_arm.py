from pathlib import Path
import time

import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# 1. Locate the project and robot model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# ---------------------------------------------------------
# 2. Load the robot
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)


print("Robot simulation loaded successfully.")
print(f"Joints: {model.njnt}")
print(f"DOF: {model.nv}")
print(f"Actuators: {model.nu}")


# ---------------------------------------------------------
# 3. Open the MuJoCo viewer
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Robot simulation started.")

    while viewer.is_running():

        # -------------------------------------------------
        # Motor commands
        # -------------------------------------------------

        data.ctrl[0] = 0.5
        data.ctrl[1] = 0.3
        data.ctrl[2] = -0.2

        # -------------------------------------------------
        # Advance physics
        # -------------------------------------------------

        mujoco.mj_step(model, data)

        # -------------------------------------------------
        # Display current joint positions
        # -------------------------------------------------

        print(
            f"\rJoint positions: "
            f"{data.qpos[0]: .3f}, "
            f"{data.qpos[1]: .3f}, "
            f"{data.qpos[2]: .3f}",
            end=""
        )

        # Update viewer
        viewer.sync()

        # Slow down visualization
        time.sleep(0.01)