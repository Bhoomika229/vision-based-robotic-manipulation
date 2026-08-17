from pathlib import Path
import time

import mujoco
import mujoco.viewer


# =========================================================
# 1. Locate MuJoCo model
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# =========================================================
# 2. Load model
# =========================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("Robot model loaded successfully.")


# =========================================================
# 3. Robot joints
# =========================================================

robot_joint_ids = [0, 1, 2]

robot_qpos_indices = [
    int(model.jnt_qposadr[joint_id])
    for joint_id in robot_joint_ids
]

print("Robot qpos indices:", robot_qpos_indices)


# =========================================================
# 4. Find gripper actuators
# =========================================================

left_gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_left_motor"
)

right_gripper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_right_motor"
)

print("Left gripper actuator:", left_gripper_id)
print("Right gripper actuator:", right_gripper_id)


# =========================================================
# 5. Find the ACTUAL end-effector geometry
# =========================================================

end_effector_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "end_effector"
)

cube_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "target_object_geom"
)

print("End-effector geom ID:", end_effector_geom_id)
print("Cube geom ID:", cube_geom_id)


# =========================================================
# 6. Check IDs
# =========================================================

if end_effector_geom_id < 0:
    raise RuntimeError(
        "Could not find geom named 'end_effector'."
    )

if cube_geom_id < 0:
    raise RuntimeError(
        "Could not find geom named 'target_object_geom'."
    )


# =========================================================
# 7. Get REAL gripper position
# =========================================================

def get_gripper_position():

    # data.geom_xpos gives the WORLD position of the
    # actual end-effector sphere.
    return data.geom_xpos[end_effector_geom_id].copy()


# =========================================================
# 8. Get cube position
# =========================================================

def get_cube_position():

    return data.geom_xpos[cube_geom_id].copy()


# =========================================================
# 9. Calculate distance
# =========================================================

def calculate_distance():

    gripper = get_gripper_position()
    cube = get_cube_position()

    dx = gripper[0] - cube[0]
    dy = gripper[1] - cube[1]
    dz = gripper[2] - cube[2]

    return (dx * dx + dy * dy + dz * dz) ** 0.5


# =========================================================
# 10. Print REAL positions
# =========================================================

def print_positions(title):

    gripper = get_gripper_position()
    cube = get_cube_position()
    distance = calculate_distance()

    print("\n========================================")
    print(title)
    print("========================================")

    print(
        f"REAL GRIPPER: "
        f"X={gripper[0]:.3f} "
        f"Y={gripper[1]:.3f} "
        f"Z={gripper[2]:.3f}"
    )

    print(
        f"CUBE: "
        f"X={cube[0]:.3f} "
        f"Y={cube[1]:.3f} "
        f"Z={cube[2]:.3f}"
    )

    print(
        f"GRIPPER-CUBE DISTANCE: "
        f"{distance:.3f} m"
    )

    print("========================================")


# =========================================================
# 11. Smooth robot movement
# =========================================================

def move_joints(viewer, target_positions, duration=4.0):

    start_positions = [
        float(data.qpos[index])
        for index in robot_qpos_indices
    ]

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        progress = min(elapsed / duration, 1.0)

        # Smoothstep interpolation
        smooth_progress = (
            progress * progress *
            (3.0 - 2.0 * progress)
        )

        for i, qpos_index in enumerate(robot_qpos_indices):

            start = start_positions[i]
            target = target_positions[i]

            data.qpos[qpos_index] = (
                start
                + smooth_progress * (target - start)
            )

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:
            break


# =========================================================
# 12. Open gripper
# =========================================================

def open_gripper(viewer, duration=1.5):

    print("\nOpening gripper...")

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        progress = min(elapsed / duration, 1.0)

        if left_gripper_id >= 0:
            data.ctrl[left_gripper_id] = 1.0

        if right_gripper_id >= 0:
            data.ctrl[right_gripper_id] = 1.0

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:
            break

    if left_gripper_id >= 0:
        data.ctrl[left_gripper_id] = 0

    if right_gripper_id >= 0:
        data.ctrl[right_gripper_id] = 0

    print("Gripper opened.")


# =========================================================
# 13. Close gripper
# =========================================================

def close_gripper(viewer, duration=1.5):

    print("\nClosing gripper...")

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        progress = min(elapsed / duration, 1.0)

        if left_gripper_id >= 0:
            data.ctrl[left_gripper_id] = -1.0

        if right_gripper_id >= 0:
            data.ctrl[right_gripper_id] = -1.0

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:
            break

    if left_gripper_id >= 0:
        data.ctrl[left_gripper_id] = 0

    if right_gripper_id >= 0:
        data.ctrl[right_gripper_id] = 0

    print("Gripper closed.")


# =========================================================
# 14. KNOWN-GOOD configurations
# =========================================================

# DO NOT CHANGE THESE.
# These are the configurations that previously
# brought the robot visually near the cube.

approach_position = [
    -0.4482,
    -1.3461,
    2.7532
]

grasp_position = [
    -0.6855,
    -0.8450,
    2.6778
]


# =========================================================
# 15. Start viewer
# =========================================================

print("\n========================================")
print("       ROBOT GRASP DIAGNOSTIC")
print("========================================")

print("\nStarting MuJoCo viewer...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    # -----------------------------------------------------
    # Viewer camera
    # -----------------------------------------------------

    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 3.0


    # =====================================================
    # STEP 1 — Open gripper
    # =====================================================

    open_gripper(viewer)

    time.sleep(0.5)


    # =====================================================
    # STEP 2 — Approach
    # =====================================================

    print("\nSTEP 1: Moving to known-good approach position...")

    move_joints(
        viewer,
        approach_position,
        duration=4.0
    )

    print_positions(
        "REAL POSITION AT APPROACH"
    )

    time.sleep(1.0)


    # =====================================================
    # STEP 3 — Grasp configuration
    # =====================================================

    print("\nSTEP 2: Moving to known-good grasp position...")

    move_joints(
        viewer,
        grasp_position,
        duration=4.0
    )

    print_positions(
        "REAL POSITION AT GRASP"
    )

    time.sleep(1.0)


    # =====================================================
    # STEP 4 — Close gripper
    # =====================================================

    print("\nSTEP 3: Closing gripper...")

    close_gripper(viewer)

    time.sleep(1.0)

    print_positions(
        "REAL POSITION AFTER GRIPPER CLOSED"
    )


    # =====================================================
    # STEP 5 — Hold
    # =====================================================

    print("\nHolding grasp for 2 seconds...")

    hold_start = time.time()

    while viewer.is_running():

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.02)

        if time.time() - hold_start >= 2.0:
            break


    # =====================================================
    # STEP 6 — DO NOT LIFT YET
    # =====================================================

    print("\n========================================")
    print("          GRASP DIAGNOSTIC COMPLETE")
    print("========================================")

    print("\nWe are NOT lifting the robot yet.")

    print(
        "\nThe important value is:"
        "\nGRIPPER-CUBE DISTANCE"
    )

    print(
        "\nThis tells us whether the actual"
        "\nred gripper sphere is really at"
        "\nthe cube."
    )

    print(
        "\nIf the distance is small, we will"
        "\nimplement the real pick-up."
    )

    print(
        "\nIf the distance is large, we will"
        "\nadjust only the grasp configuration."
    )

    print("\nClose the MuJoCo window when finished.")

    while viewer.is_running():

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.02)


print("\nRobot grasp diagnostic completed.")