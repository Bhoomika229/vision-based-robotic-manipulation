from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer


# ============================================================
# 1. Locate MuJoCo model
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# ============================================================
# 2. Load model
# ============================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

print("Robot model loaded successfully.")


# ============================================================
# 3. Robot arm joints
# ============================================================

robot_joint_ids = [0, 1, 2]

robot_qpos_indices = [
    int(model.jnt_qposadr[joint_id])
    for joint_id in robot_joint_ids
]

print("Robot arm qpos indices:", robot_qpos_indices)


# ============================================================
# 4. Find important model elements
# ============================================================

end_effector_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link3"
)

cube_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target_object"
)

end_effector_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "end_effector"
)

left_finger_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "gripper_left_finger"
)

right_finger_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "gripper_right_finger"
)

cube_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "target_object_geom"
)

vertical_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_vertical_joint"
)

left_finger_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_left_joint"
)

right_finger_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_right_joint"
)


print("\n========================================")
print("       CORRECTED GRASP ALIGNMENT TEST")
print("========================================")

print("End-effector body ID:", end_effector_body_id)
print("Cube body ID:", cube_body_id)
print("End-effector geom ID:", end_effector_geom_id)
print("Left finger geom ID:", left_finger_geom_id)
print("Right finger geom ID:", right_finger_geom_id)
print("Cube geom ID:", cube_geom_id)

print("Vertical joint ID:", vertical_joint_id)
print("Left finger joint ID:", left_finger_joint_id)
print("Right finger joint ID:", right_finger_joint_id)


# ============================================================
# 5. QPOS addresses
# ============================================================

vertical_qpos = int(model.jnt_qposadr[vertical_joint_id])
left_finger_qpos = int(model.jnt_qposadr[left_finger_joint_id])
right_finger_qpos = int(model.jnt_qposadr[right_finger_joint_id])

print("\nQPOS addresses:")
print("Vertical:", vertical_qpos)
print("Left finger:", left_finger_qpos)
print("Right finger:", right_finger_qpos)


# ============================================================
# 6. IMPORTANT: known-good arm configuration
# ============================================================

approach_position = [
    -0.4482,
    -1.3461,
    2.7532
]


# ------------------------------------------------------------
# OLD grasp position was:
#
# [-0.6855, -0.8450, 2.6778]
#
# It produced:
#
# X = 0.853
# Y = -0.750
#
# which is approximately 5 cm away from the cube in X/Y.
#
# NEW grasp position:
#
# [-0.4889, -1.5249, 2.6778]
#
# This was calculated to produce:
#
# X = 0.800
# Y = -0.700
#
# exactly matching the cube XY position.
# ------------------------------------------------------------

corrected_grasp_position = [
    -0.4889,
    -1.5249,
    2.6778
]


# ============================================================
# 7. Cube target
# ============================================================

TARGET_X = 0.800
TARGET_Y = -0.700


# ============================================================
# 8. Helper functions
# ============================================================

def get_body_position(body_id):

    return data.xpos[body_id].copy()


def get_geom_position(geom_id):

    return data.geom_xpos[geom_id].copy()


def distance(a, b):

    return float(math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    ))


def print_positions(title):

    mujoco.mj_forward(model, data)

    end_effector = get_geom_position(
        end_effector_geom_id
    )

    left_finger = get_geom_position(
        left_finger_geom_id
    )

    right_finger = get_geom_position(
        right_finger_geom_id
    )

    cube = get_body_position(
        cube_body_id
    )

    midpoint = (
        left_finger + right_finger
    ) / 2.0

    print("\n========================================")
    print(title)
    print("========================================")

    print(
        f"END EFFECTOR : "
        f"X={end_effector[0]:.3f} "
        f"Y={end_effector[1]:.3f} "
        f"Z={end_effector[2]:.3f}"
    )

    print(
        f"LEFT FINGER  : "
        f"X={left_finger[0]:.3f} "
        f"Y={left_finger[1]:.3f} "
        f"Z={left_finger[2]:.3f}"
    )

    print(
        f"RIGHT FINGER : "
        f"X={right_finger[0]:.3f} "
        f"Y={right_finger[1]:.3f} "
        f"Z={right_finger[2]:.3f}"
    )

    print(
        f"FINGER MIDPT : "
        f"X={midpoint[0]:.3f} "
        f"Y={midpoint[1]:.3f} "
        f"Z={midpoint[2]:.3f}"
    )

    print(
        f"CUBE         : "
        f"X={cube[0]:.3f} "
        f"Y={cube[1]:.3f} "
        f"Z={cube[2]:.3f}"
    )

    print("\nDistances:")

    print(
        f"Left finger -> cube : "
        f"{distance(left_finger, cube):.3f} m"
    )

    print(
        f"Right finger -> cube: "
        f"{distance(right_finger, cube):.3f} m"
    )

    print(
        f"Midpoint -> cube     : "
        f"{distance(midpoint, cube):.3f} m"
    )

    print(
        f"End effector -> cube : "
        f"{distance(end_effector, cube):.3f} m"
    )

    print("\nJoint positions:")

    print(
        f"Vertical gripper: "
        f"{data.qpos[vertical_qpos]:.3f}"
    )

    print(
        f"Left finger: "
        f"{data.qpos[left_finger_qpos]:.3f}"
    )

    print(
        f"Right finger: "
        f"{data.qpos[right_finger_qpos]:.3f}"
    )

    print("========================================")


def move_arm(viewer, target_positions, duration=4.0):

    start_positions = [
        float(data.qpos[index])
        for index in robot_qpos_indices
    ]

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        progress = min(
            elapsed / duration,
            1.0
        )

        smooth_progress = (
            progress
            * progress
            * (3.0 - 2.0 * progress)
        )

        for i, qpos_index in enumerate(
            robot_qpos_indices
        ):

            start = start_positions[i]

            target = target_positions[i]

            data.qpos[qpos_index] = (
                start
                + smooth_progress
                * (target - start)
            )

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:

            break


def set_fingers(viewer, position):

    data.qpos[left_finger_qpos] = position
    data.qpos[right_finger_qpos] = position

    mujoco.mj_forward(model, data)

    viewer.sync()


# ============================================================
# 9. Start viewer
# ============================================================

print("\nStarting MuJoCo viewer...")

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 3.0


    # ========================================================
    # STEP 1
    # Open fingers
    # ========================================================

    print("\nSTEP 1: Opening fingers...")

    set_fingers(
        viewer,
        0.0
    )

    time.sleep(1.0)

    print_positions(
        "FINGERS OPEN"
    )


    # ========================================================
    # STEP 2
    # Move to known-good approach
    # ========================================================

    print(
        "\nSTEP 2: Moving to known-good approach position..."
    )

    move_arm(
        viewer,
        approach_position,
        duration=4.0
    )

    time.sleep(0.5)

    print_positions(
        "ROBOT AT APPROACH"
    )


    # ========================================================
    # STEP 3
    # Move to CORRECTED grasp position
    # ========================================================

    print(
        "\nSTEP 3: Moving to CORRECTED grasp position..."
    )

    print("\nOld grasp XY:")
    print("X = 0.853")
    print("Y = -0.750")

    print("\nTarget cube XY:")
    print("X = 0.800")
    print("Y = -0.700")

    print("\nUsing corrected joint configuration:")

    for i, angle in enumerate(
        corrected_grasp_position
    ):

        print(
            f"Joint {i + 1}: "
            f"{angle:.4f} rad"
        )

    move_arm(
        viewer,
        corrected_grasp_position,
        duration=4.0
    )

    time.sleep(1.0)

    print_positions(
        "CORRECTED GRASP POSITION"
    )


    # ========================================================
    # STEP 4
    # Keep vertical gripper at zero
    # ========================================================

    print(
        "\nSTEP 4: Keeping vertical gripper at 0.000..."
    )

    data.qpos[vertical_qpos] = 0.0

    mujoco.mj_forward(model, data)

    viewer.sync()

    time.sleep(0.5)

    print_positions(
        "READY FOR FINGER CLOSURE"
    )


    # ========================================================
    # STEP 5
    # Close fingers DIRECTLY
    #
    # IMPORTANT:
    # We DO NOT call mj_step here.
    #
    # This prevents the arm motors from accidentally
    # moving the robot while testing the fingers.
    # ========================================================

    print(
        "\nSTEP 5: Closing fingers DIRECTLY..."
    )

    print(
        "\nFinger movement:"
    )

    print(
        "Left finger : 0.000 -> 0.075"
    )

    print(
        "Right finger: 0.000 -> 0.075"
    )

    set_fingers(
        viewer,
        0.075
    )

    time.sleep(1.0)

    print_positions(
        "AFTER DIRECT FINGER CLOSURE"
    )


    # ========================================================
    # STEP 6
    # Final alignment evaluation
    # ========================================================

    mujoco.mj_forward(model, data)

    left_finger = get_geom_position(
        left_finger_geom_id
    )

    right_finger = get_geom_position(
        right_finger_geom_id
    )

    cube = get_body_position(
        cube_body_id
    )

    midpoint = (
        left_finger + right_finger
    ) / 2.0

    midpoint_distance = distance(
        midpoint,
        cube
    )


    print("\n========================================")
    print("       ALIGNMENT RESULT")
    print("========================================")

    print(
        f"Finger midpoint X: {midpoint[0]:.3f}"
    )

    print(
        f"Finger midpoint Y: {midpoint[1]:.3f}"
    )

    print(
        f"Finger midpoint Z: {midpoint[2]:.3f}"
    )

    print(
        f"Cube X: {cube[0]:.3f}"
    )

    print(
        f"Cube Y: {cube[1]:.3f}"
    )

    print(
        f"Cube Z: {cube[2]:.3f}"
    )

    print(
        f"\nMidpoint -> Cube: "
        f"{midpoint_distance:.3f} m"
    )

    print(
        "\nThe target is for the cube to be "
        "between the two fingers."
    )

    print(
        "\nDO NOT LIFT YET."
    )

    print(
        "Close the MuJoCo window when finished."
    )


    # ========================================================
    # STEP 7
    # Keep viewer open
    # ========================================================

    while viewer.is_running():

        mujoco.mj_forward(
            model,
            data
        )

        viewer.sync()

        time.sleep(0.02)


print(
    "\nCorrected gripper alignment test completed."
)