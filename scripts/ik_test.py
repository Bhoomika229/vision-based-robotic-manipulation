from pathlib import Path
import mujoco
import numpy as np


# =========================================================
# 1. Load robot model
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Robot model loaded successfully.")


# =========================================================
# 2. Target position detected by the camera
# =========================================================

target = np.array([
    0.853,
    -0.750,
    0.250
])

print()
print("Target position from camera:")
print(f"X = {target[0]:.3f} m")
print(f"Y = {target[1]:.3f} m")
print(f"Z = {target[2]:.3f} m")


# =========================================================
# 3. Identify the robot's three actuated joints
# =========================================================

robot_joint_ids = []

for actuator_id in range(model.nu):

    joint_id = model.actuator_trnid[actuator_id, 0]

    robot_joint_ids.append(int(joint_id))


print()
print("Robot joint IDs:", robot_joint_ids)


# =========================================================
# 4. Get qpos and DOF addresses
# =========================================================

robot_qpos_indices = []
robot_dof_indices = []

for joint_id in robot_joint_ids:

    qpos_address = model.jnt_qposadr[joint_id]
    dof_address = model.jnt_dofadr[joint_id]

    robot_qpos_indices.append(int(qpos_address))
    robot_dof_indices.append(int(dof_address))


print("Robot qpos indices:", robot_qpos_indices)
print("Robot DOF indices:", robot_dof_indices)


# =========================================================
# 5. Select the ACTUAL robot end-effector
# =========================================================
#
# Body structure:
#
# Body 0 = world
# Body 1 = base
# Body 2 = link1
# Body 3 = link2
# Body 4 = link3
# Body 5 = target_object
#
# Therefore link3 is the robot end-effector.
# =========================================================

end_effector_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "link3"
)

print()
print("End-effector body ID:", end_effector_id)


# =========================================================
# 6. Initialize simulation
# =========================================================

mujoco.mj_forward(model, data)

current_position = data.xpos[
    end_effector_id
].copy()

print()
print("Current robot end-effector position:")
print(
    f"X = {current_position[0]:.3f} m, "
    f"Y = {current_position[1]:.3f} m, "
    f"Z = {current_position[2]:.3f} m"
)


# =========================================================
# 7. Calculate distance from robot to target
# =========================================================

distance = np.linalg.norm(
    target - current_position
)

print()
print(f"Distance to target: {distance:.3f} m")


# =========================================================
# 8. Create Jacobian arrays
# =========================================================

jacp = np.zeros((3, model.nv))

jacr = np.zeros((3, model.nv))


# =========================================================
# 9. IK parameters
# =========================================================

learning_rate = 0.5

max_iterations = 1000

tolerance = 0.005


# =========================================================
# 10. Start inverse kinematics
# =========================================================

print()
print("Starting inverse kinematics...")


converged = False


for iteration in range(max_iterations):

    # -----------------------------------------------------
    # Update forward kinematics
    # -----------------------------------------------------

    mujoco.mj_forward(model, data)


    # -----------------------------------------------------
    # Current end-effector position
    # -----------------------------------------------------

    current_position = data.xpos[
        end_effector_id
    ].copy()


    # -----------------------------------------------------
    # Calculate position error
    # -----------------------------------------------------

    error = target - current_position

    error_norm = np.linalg.norm(error)


    # -----------------------------------------------------
    # Check convergence
    # -----------------------------------------------------

    if error_norm < tolerance:

        converged = True

        print()
        print("IK converged successfully!")

        print(
            f"Iterations: {iteration}"
        )

        print(
            f"Final error: {error_norm:.4f} m"
        )

        break


    # -----------------------------------------------------
    # Calculate full robot Jacobian
    # -----------------------------------------------------

    mujoco.mj_jacBody(
        model,
        data,
        jacp,
        jacr,
        end_effector_id
    )


    # -----------------------------------------------------
    # Select ONLY the three robot joint columns
    # -----------------------------------------------------

    robot_jacobian = jacp[
        :,
        robot_dof_indices
    ]


    # -----------------------------------------------------
    # Damped least-squares inverse
    # -----------------------------------------------------

    damping = 0.01

    J = robot_jacobian

    JJT = J @ J.T

    dq = (
        J.T
        @ np.linalg.inv(
            JJT + damping * np.eye(3)
        )
        @ error
    )


    # -----------------------------------------------------
    # Limit joint movement per iteration
    # -----------------------------------------------------

    dq = np.clip(
        dq,
        -0.05,
        0.05
    )


    # -----------------------------------------------------
    # Update ONLY robot joint positions
    # -----------------------------------------------------

    for i, qpos_index in enumerate(
        robot_qpos_indices
    ):

        data.qpos[qpos_index] += (
            learning_rate * dq[i]
        )


    # -----------------------------------------------------
    # Apply joint limits
    # -----------------------------------------------------

    for joint_id in robot_joint_ids:

        if model.jnt_limited[joint_id]:

            low, high = model.jnt_range[joint_id]

            qpos_index = model.jnt_qposadr[
                joint_id
            ]

            data.qpos[qpos_index] = np.clip(
                data.qpos[qpos_index],
                low,
                high
            )


# =========================================================
# 11. Handle failure to converge
# =========================================================

if not converged:

    print()
    print("IK did not fully converge.")

    print(
        f"Final position error: "
        f"{error_norm:.4f} m"
    )


# =========================================================
# 12. Final forward kinematics
# =========================================================

mujoco.mj_forward(
    model,
    data
)

final_position = data.xpos[
    end_effector_id
].copy()


# =========================================================
# 13. Print final IK result
# =========================================================

print()
print("========================================")
print("              IK RESULT")
print("========================================")


print()
print("Target position from vision:")

print(
    f"X = {target[0]:.3f} m"
)

print(
    f"Y = {target[1]:.3f} m"
)

print(
    f"Z = {target[2]:.3f} m"
)


print()
print("Final robot end-effector position:")

print(
    f"X = {final_position[0]:.3f} m"
)

print(
    f"Y = {final_position[1]:.3f} m"
)

print(
    f"Z = {final_position[2]:.3f} m"
)


print()
print("Final position error:")

final_error = np.linalg.norm(
    target - final_position
)

print(
    f"{final_error:.4f} m"
)


# =========================================================
# 14. Print robot joint angles
# =========================================================

print()
print("Robot joint positions:")

for i, qpos_index in enumerate(
    robot_qpos_indices
):

    joint_angle = data.qpos[
        qpos_index
    ]

    print(
        f"Joint {i + 1}: "
        f"{joint_angle:.4f} rad"
    )


# =========================================================
# 15. Finish
# =========================================================

print()
print("========================================")
print("IK test completed.")
print("========================================")