from pathlib import Path
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

print("========== MODEL BODIES ==========")

for body_id in range(model.nbody):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    print(
        f"Body ID {body_id}: {name}"
    )

print("===================================")