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

mujoco.mj_forward(model, data)

print("Robot model loaded successfully.")


# =========================================================
# 3. Find vertical gripper joint
# =========================================================

vertical_joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "gripper_vertical_joint"
)

vertical_qpos = int(model.jnt_qposadr[vertical_joint_id])

vertical_actuator_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "gripper_vertical_motor"
)

print("Vertical gripper joint ID:", vertical_joint_id)
print("Vertical gripper qpos address:", vertical_qpos)
print("Vertical gripper actuator ID:", vertical_actuator_id)


# =========================================================
# 4. Helper function
# =========================================================

def move_vertical(viewer, target, duration=2.0):

    start = float(data.qpos[vertical_qpos])

    print(
        f"\nMoving vertical gripper:"
        f" {start:.3f} -> {target:.3f}"
    )

    start_time = time.time()

    while viewer.is_running():

        elapsed = time.time() - start_time

        progress = min(elapsed / duration, 1.0)

        # Smooth movement
        smooth = progress * progress * (3.0 - 2.0 * progress)

        data.qpos[vertical_qpos] = (
            start + smooth * (target - start)
        )

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.01)

        if progress >= 1.0:
            break


# =========================================================
# 5. Start viewer
# =========================================================

print("\nStarting vertical gripper test...")

with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    viewer.cam.distance = 3.0

    # =====================================================
    # Initial position
    # =====================================================

    data.qpos[vertical_qpos] = 0.0

    mujoco.mj_forward(model, data)

    viewer.sync()

    print("\nInitial vertical position:")
    print(f"Z slide = {data.qpos[vertical_qpos]:.3f}")

    time.sleep(1.0)


    # =====================================================
    # STEP 1 — Move DOWN
    # =====================================================

    print("\nSTEP 1: Moving gripper DOWN...")

    move_vertical(
        viewer,
        target=-0.15,
        duration=2.0
    )

    print("Gripper moved DOWN.")

    time.sleep(1.0)


    # =====================================================
    # STEP 2 — Move further DOWN
    # =====================================================

    print("\nSTEP 2: Moving gripper further DOWN...")

    move_vertical(
        viewer,
        target=-0.22,
        duration=2.0
    )

    print("Gripper reached lower position.")

    time.sleep(1.0)


    # =====================================================
    # STEP 3 — Move UP
    # =====================================================

    print("\nSTEP 3: Moving gripper UP...")

    move_vertical(
        viewer,
        target=0.0,
        duration=2.0
    )

    print("Gripper returned to original position.")

    time.sleep(1.0)


    # =====================================================
    # Keep viewer open
    # =====================================================

    print("\n========================================")
    print("VERTICAL GRIPPER TEST COMPLETE")
    print("========================================")

    print("\nThe viewer will remain open.")
    print("Close the MuJoCo window when finished.")

    while viewer.is_running():

        mujoco.mj_forward(model, data)

        viewer.sync()

        time.sleep(0.02)


print("\nVertical gripper test completed.")