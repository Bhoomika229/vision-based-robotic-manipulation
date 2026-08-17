from pathlib import Path
import time

import mujoco
import mujoco.viewer


# =========================================================
# 1. Locate model
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# =========================================================
# 2. Load model
# =========================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Robot model loaded successfully.")


# =========================================================
# 3. Find gripper joints
# =========================================================

left_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_left_joint"
)

right_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_right_joint"
)


# Get qpos addresses
left_qpos = model.jnt_qposadr[left_joint_id]
right_qpos = model.jnt_qposadr[right_joint_id]


print("Left gripper joint ID:", left_joint_id)
print("Right gripper joint ID:", right_joint_id)

print("Left qpos address:", left_qpos)
print("Right qpos address:", right_qpos)


# =========================================================
# 4. Find gripper actuators
# =========================================================

left_motor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_left_motor"
)

right_motor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_right_motor"
)


print("Left gripper actuator:", left_motor_id)
print("Right gripper actuator:", right_motor_id)


# =========================================================
# 5. Open viewer
# =========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\nGripper test started.")
    print("Opening gripper...")

    # -----------------------------------------------------
    # OPEN GRIPPER
    # -----------------------------------------------------

    for step in range(1500):

        data.ctrl[left_motor_id] = 1.0
        data.ctrl[right_motor_id] = 1.0

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)


    print("Gripper opened.")


    time.sleep(1)


    # -----------------------------------------------------
    # CLOSE GRIPPER
    # -----------------------------------------------------

    print("Closing gripper...")

    for step in range(1500):

        data.ctrl[left_motor_id] = -1.0
        data.ctrl[right_motor_id] = -1.0

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)


    print("Gripper closed.")


    time.sleep(2)


    # -----------------------------------------------------
    # STOP MOTORS
    # -----------------------------------------------------

    data.ctrl[left_motor_id] = 0
    data.ctrl[right_motor_id] = 0

    print("Gripper test completed.")

    time.sleep(2)