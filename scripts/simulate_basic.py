from pathlib import Path
import time
import mujoco
import mujoco.viewer


# ---------------------------------------------------------
# 1. Locate the project
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "basic_world.xml"


# ---------------------------------------------------------
# 2. Load the MuJoCo model
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

# Create the simulation state
data = mujoco.MjData(model)


print("MuJoCo model loaded successfully.")
print(f"Simulation timestep: {model.opt.timestep} seconds")
print(f"Number of bodies: {model.nbody}")
print(f"Number of degrees of freedom: {model.nv}")


# ---------------------------------------------------------
# 3. Launch the viewer
# ---------------------------------------------------------

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("Simulation started.")

    while viewer.is_running():

        # Advance the physics simulation
        mujoco.mj_step(model, data)

        # Print the ball's height
        print(f"\rBall Z position: {data.qpos[2]:.3f} m", end="")

        # Update the viewer
        viewer.sync()

        # Slow the simulation down so we can observe it
        time.sleep(0.02)