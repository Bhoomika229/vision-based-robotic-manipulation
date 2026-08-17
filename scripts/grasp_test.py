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
# 3. Find robot joints
# =========================================================

joint_names = [
    "joint1",
    "joint2",
    "joint3"
]

joint_ids = []

for name in joint_names:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )
    joint_ids.append(joint_id)

print("Robot joint IDs:", joint_ids)


# =========================================================
# 4. Find gripper actuators
# =========================================================

left_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_left_motor"
)

right_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_right_motor"
)

print("Left gripper actuator:", left_motor)
print("Right gripper actuator:", right_motor)


# =========================================================
# 5. Find target cube
# =========================================================

cube_body = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target_object"
)

print("Target cube body ID:", cube_body)


# =========================================================
# 6. Start viewer
# =========================================================

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("\n========================================")
    print("GRASP TEST STARTED")
    print("========================================")

    # -----------------------------------------------------
    # STEP 1 — Open gripper
    # -----------------------------------------------------

    print("\n1. Opening gripper...")

    for _ in range(1000):

        data.ctrl[left_motor] = 1.0
        data.ctrl[right_motor] = 1.0

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.002)

    data.ctrl[left_motor] = 0
    data.ctrl[right_motor] = 0

    print("Gripper opened.")

    time.sleep(1)


    # -----------------------------------------------------
    # STEP 2 — Move arm toward cube
    #
    # These are deliberately gentle joint movements.
    # -----------------------------------------------------

    print("\n2. Moving arm toward cube...")

    target_q = [
        -0.30,
        -1.10,
        0.20
    ]

    for _ in range(2500):

        for i, joint_id in enumerate(joint_ids):

            qpos_index = model.jnt_qposadr[joint_id]

            current = data.qpos[qpos_index]

            error = target_q[i] - current

            data.ctrl[i] = 8.0 * error

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.002)


    # Stop arm motors
    data.ctrl[0] = 0
    data.ctrl[1] = 0
    data.ctrl[2] = 0

    print("Arm reached grasp position.")

    time.sleep(1)


    # -----------------------------------------------------
    # STEP 3 — Close gripper
    # -----------------------------------------------------

    print("\n3. Closing gripper around cube...")

    for _ in range(1800):

        data.ctrl[left_motor] = -1.0
        data.ctrl[right_motor] = -1.0

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.002)

    data.ctrl[left_motor] = 0
    data.ctrl[right_motor] = 0

    print("Gripper closed.")

    time.sleep(2)


    # -----------------------------------------------------
    # STEP 4 — Print cube position
    # -----------------------------------------------------

    print("\nCube position after grasp:")
    print(
        "X = {:.3f} m".format(data.xpos[cube_body][0])
    )
    print(
        "Y = {:.3f} m".format(data.xpos[cube_body][1])
    )
    print(
        "Z = {:.3f} m".format(data.xpos[cube_body][2])
    )


    # -----------------------------------------------------
    # STEP 5 — Hold
    # -----------------------------------------------------

    print("\nHolding cube for 3 seconds...")

    for _ in range(1500):

        mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.002)


    print("\n========================================")
    print("GRASP TEST COMPLETED")
    print("========================================")

    time.sleep(2)