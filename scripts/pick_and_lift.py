from pathlib import Path
import time
import math

import mujoco
import mujoco.viewer


# ============================================================
# 1. Locate model
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
# 4. Find bodies
# ============================================================

cube_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "target_object"
)

end_effector_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link3"
)

left_finger_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "gripper_left"
)

right_finger_body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "gripper_right"
)


# ============================================================
# 5. Find gripper joints
# ============================================================

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


vertical_qpos = int(
    model.jnt_qposadr[vertical_joint_id]
)

left_finger_qpos = int(
    model.jnt_qposadr[left_finger_joint_id]
)

right_finger_qpos = int(
    model.jnt_qposadr[right_finger_joint_id]
)


# ============================================================
# 6. Find cube free joint
# ============================================================

cube_joint_id = -1

for joint_id in range(model.njnt):

    if model.jnt_bodyid[joint_id] == cube_body_id:

        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:

            cube_joint_id = joint_id
            break


if cube_joint_id < 0:

    raise RuntimeError(
        "Could not find cube free joint."
    )


cube_qpos = int(
    model.jnt_qposadr[cube_joint_id]
)


# ============================================================
# 7. VERIFIED GRASP CONFIGURATION
# ============================================================

# THIS CONFIGURATION WAS ALREADY VERIFIED.
# DO NOT CHANGE IT.

grasp_position = [
    0.6180,
    -1.8820,
    -0.4345
]


# ============================================================
# 8. DROP TARGET
# ============================================================

# The cube will be moved here after lifting.

DROP_X = 0.30
DROP_Y = 0.40


# ============================================================
# 9. Calculate IK for drop location
# ============================================================

def calculate_drop_ik(x, y):

    # Robot link lengths
    L1 = 0.8
    L2 = 0.7
    L3 = 0.5

    # During transport we keep joint 3 at zero.
    #
    # Therefore links 2 + 3 act as one 1.2 m segment.

    L23 = L2 + L3

    distance_squared = (
        x * x
        + y * y
    )

    cos_q2 = (
        distance_squared
        - L1 * L1
        - L23 * L23
    ) / (
        2.0 * L1 * L23
    )

    cos_q2 = max(
        -1.0,
        min(1.0, cos_q2)
    )

    q2 = math.acos(cos_q2)

    q1 = (
        math.atan2(y, x)
        - math.atan2(
            L23 * math.sin(q2),
            L1 + L23 * math.cos(q2)
        )
    )

    q3 = 0.0

    return [
        q1,
        q2,
        q3
    ]


drop_position = calculate_drop_ik(
    DROP_X,
    DROP_Y
)


# ============================================================
# 10. Print configuration
# ============================================================

print("\n========================================")
print("        PICK → LIFT → MOVE → RELEASE")
print("========================================")

print("\nVerified grasp configuration:")

print(
    f"Joint 1: {grasp_position[0]:.4f}"
)

print(
    f"Joint 2: {grasp_position[1]:.4f}"
)

print(
    f"Joint 3: {grasp_position[2]:.4f}"
)

print("\nDrop target:")

print(
    f"X = {DROP_X:.3f}"
)

print(
    f"Y = {DROP_Y:.3f}"
)

print("\nCalculated drop configuration:")

print(
    f"Joint 1: {drop_position[0]:.4f}"
)

print(
    f"Joint 2: {drop_position[1]:.4f}"
)

print(
    f"Joint 3: {drop_position[2]:.4f}"
)


# ============================================================
# 11. Cube attachment state
# ============================================================

cube_attached = False

initial_cube_position = None

initial_finger_midpoint = None


# ============================================================
# 12. Get finger midpoint
# ============================================================

def get_finger_midpoint():

    left = data.xpos[
        left_finger_body_id
    ].copy()

    right = data.xpos[
        right_finger_body_id
    ].copy()

    return (
        left + right
    ) / 2.0


# ============================================================
# 13. Attach cube
# ============================================================

def attach_cube():

    global cube_attached
    global initial_cube_position
    global initial_finger_midpoint

    mujoco.mj_forward(
        model,
        data
    )

    initial_cube_position = (
        data.xpos[
            cube_body_id
        ].copy()
    )

    initial_finger_midpoint = (
        get_finger_midpoint()
    )

    cube_attached = True

    # Stop cube velocity.

    data.qvel[
        cube_qpos:cube_qpos + 6
    ] = 0.0


# ============================================================
# 14. Update attached cube
# ============================================================

def update_cube_attachment():

    if not cube_attached:

        return

    mujoco.mj_forward(
        model,
        data
    )

    current_midpoint = (
        get_finger_midpoint()
    )

    displacement = (
        current_midpoint
        - initial_finger_midpoint
    )

    new_cube_position = (
        initial_cube_position
        + displacement
    )

    # Move cube with gripper.

    data.qpos[
        cube_qpos:cube_qpos + 3
    ] = new_cube_position

    # Keep cube orientation fixed.

    data.qvel[
        cube_qpos:cube_qpos + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# 15. Release cube
# ============================================================

def release_cube():

    global cube_attached

    cube_attached = False

    # Allow cube to behave normally under gravity.

    data.qvel[
        cube_qpos:cube_qpos + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# 16. Move arm
# ============================================================

def move_arm(
    viewer,
    target_positions,
    duration=4.0
):

    start_positions = [
        float(
            data.qpos[index]
        )
        for index in robot_qpos_indices
    ]

    start_time = time.time()

    while viewer.is_running():

        elapsed = (
            time.time()
            - start_time
        )

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

            data.qpos[
                qpos_index
            ] = (
                start
                + smooth_progress
                * (target - start)
            )

        mujoco.mj_forward(
            model,
            data
        )

        # IMPORTANT:
        # If cube is attached, it follows
        # the gripper while the arm moves.

        update_cube_attachment()

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:

            break


# ============================================================
# 17. Move vertical gripper
# ============================================================

def move_vertical(
    viewer,
    target,
    duration=3.0
):

    start = float(
        data.qpos[
            vertical_qpos
        ]
    )

    start_time = time.time()

    while viewer.is_running():

        elapsed = (
            time.time()
            - start_time
        )

        progress = min(
            elapsed / duration,
            1.0
        )

        smooth_progress = (
            progress
            * progress
            * (3.0 - 2.0 * progress)
        )

        current = (
            start
            + smooth_progress
            * (target - start)
        )

        data.qpos[
            vertical_qpos
        ] = current

        mujoco.mj_forward(
            model,
            data
        )

        update_cube_attachment()

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:

            break


# ============================================================
# 18. Set fingers
# ============================================================

def set_fingers(
    viewer,
    position
):

    data.qpos[
        left_finger_qpos
    ] = position

    data.qpos[
        right_finger_qpos
    ] = position

    mujoco.mj_forward(
        model,
        data
    )

    viewer.sync()


# ============================================================
# 19. Print state
# ============================================================

def print_state(title):

    mujoco.mj_forward(
        model,
        data
    )

    ee = data.xpos[
        end_effector_body_id
    ]

    cube = data.xpos[
        cube_body_id
    ]

    midpoint = get_finger_midpoint()

    print("\n========================================")
    print(title)
    print("========================================")

    print(
        f"END EFFECTOR : "
        f"X={ee[0]:.3f} "
        f"Y={ee[1]:.3f} "
        f"Z={ee[2]:.3f}"
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


# ============================================================
# 20. Start viewer
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 3.0


    # ========================================================
    # STEP 1 — Reset
    # ========================================================

    print(
        "\nSTEP 1: Resetting gripper..."
    )

    data.qpos[
        vertical_qpos
    ] = 0.0

    set_fingers(
        viewer,
        0.0
    )

    time.sleep(1.0)


    # ========================================================
    # STEP 2 — Move to verified grasp
    # ========================================================

    print(
        "\nSTEP 2: Moving to VERIFIED GRASP..."
    )

    move_arm(
        viewer,
        grasp_position,
        duration=4.0
    )

    time.sleep(1.0)

    print_state(
        "AT VERIFIED GRASP POSITION"
    )


    # ========================================================
    # STEP 3 — Close fingers
    # ========================================================

    print(
        "\nSTEP 3: Closing fingers..."
    )

    set_fingers(
        viewer,
        0.075
    )

    time.sleep(1.0)

    print_state(
        "CUBE GRASPED"
    )


    # ========================================================
    # STEP 4 — Attach cube
    # ========================================================

    print(
        "\nSTEP 4: Attaching cube..."
    )

    attach_cube()

    print(
        "Cube attached to gripper."
    )

    print_state(
        "CUBE ATTACHED"
    )


    # ========================================================
    # STEP 5 — LIFT
    # ========================================================

    print(
        "\nSTEP 5: LIFTING CUBE..."
    )

    move_vertical(
        viewer,
        0.080,
        duration=3.0
    )

    time.sleep(0.5)

    print_state(
        "CUBE LIFTED"
    )


    # ========================================================
    # STEP 6 — MOVE TO DROP LOCATION
    # ========================================================

    print(
        "\nSTEP 6: MOVING TO DROP LOCATION..."
    )

    print(
        f"Target X = {DROP_X:.3f}"
    )

    print(
        f"Target Y = {DROP_Y:.3f}"
    )

    print(
        "\nCube remains attached during movement."
    )

    move_arm(
        viewer,
        drop_position,
        duration=5.0
    )

    time.sleep(1.0)

    print_state(
        "ARRIVED AT DROP LOCATION"
    )


    # ========================================================
    # STEP 7 — LOWER CUBE
    # ========================================================

    print(
        "\nSTEP 7: LOWERING CUBE..."
    )

    print(
        "Lowering cube toward ground..."
    )

    # Starting vertical = +0.080
    #
    # Target vertical = -0.130
    #
    # This lowers cube approximately 0.210 m.
    #
    # Cube starts around Z=0.330 after lift.
    # Final cube should be around Z=0.120,
    # which is approximately ground contact.

    move_vertical(
        viewer,
        -0.130,
        duration=4.0
    )

    time.sleep(1.0)

    print_state(
        "CUBE AT DROP HEIGHT"
    )


    # ========================================================
    # STEP 8 — RELEASE
    # ========================================================

    print(
        "\nSTEP 8: RELEASING CUBE..."
    )

    release_cube()

    print(
        "Cube attachment released."
    )

    time.sleep(2.0)

    mujoco.mj_forward(
        model,
        data
    )

    print_state(
        "AFTER RELEASE"
    )


    # ========================================================
    # STEP 9 — OPEN FINGERS
    # ========================================================

    print(
        "\nSTEP 9: Opening fingers..."
    )

    set_fingers(
        viewer,
        0.0
    )

    time.sleep(1.0)


    # ========================================================
    # STEP 10 — Raise gripper away
    # ========================================================

    print(
        "\nSTEP 10: Raising gripper away from cube..."
    )

    move_vertical(
        viewer,
        0.0,
        duration=2.0
    )

    time.sleep(1.0)

    print_state(
        "ROBOT AFTER RELEASE"
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    cube = data.xpos[
        cube_body_id
    ]

    print("\n========================================")
    print("          FINAL RESULT")
    print("========================================")

    print(
        f"Cube X: {cube[0]:.3f} m"
    )

    print(
        f"Cube Y: {cube[1]:.3f} m"
    )

    print(
        f"Cube Z: {cube[2]:.3f} m"
    )

    print(
        f"\nTarget X: {DROP_X:.3f} m"
    )

    print(
        f"Target Y: {DROP_Y:.3f} m"
    )

    print(
        "\nPICK → LIFT → MOVE → RELEASE sequence complete."
    )

    print(
        "\nMuJoCo window will remain open."
    )

    print(
        "Close it when finished."
    )


    # ========================================================
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
    "\nRobot pick, transport and release completed."
)