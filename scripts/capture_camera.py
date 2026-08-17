from pathlib import Path

import mujoco
from PIL import Image


# ---------------------------------------------------------
# 1. Locate the MuJoCo model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "robot_arm.xml"


# ---------------------------------------------------------
# 2. Load model and simulation data
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Initialize the simulation state
mujoco.mj_forward(model, data)

print("Robot model loaded successfully.")


# ---------------------------------------------------------
# 3. Create an OpenGL context for off-screen rendering
# ---------------------------------------------------------

gl_context = mujoco.GLContext(640, 480)
gl_context.make_current()

print("OpenGL rendering context created.")


# ---------------------------------------------------------
# 4. Create MuJoCo renderer
# ---------------------------------------------------------

renderer = mujoco.Renderer(
    model,
    height=480,
    width=640
)


# ---------------------------------------------------------
# 5. Render from workspace camera
# ---------------------------------------------------------

renderer.update_scene(
    data,
    camera="workspace_camera"
)

pixels = renderer.render()


# ---------------------------------------------------------
# 6. Check rendered pixels
# ---------------------------------------------------------

print("Pixel minimum:", pixels.min())
print("Pixel maximum:", pixels.max())


# ---------------------------------------------------------
# 7. Save RGB image
# ---------------------------------------------------------

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "workspace_camera.png"

Image.fromarray(pixels).save(output_path)

print("RGB image captured successfully.")
print(f"Saved to: {output_path}")


# ---------------------------------------------------------
# 8. Clean up
# ---------------------------------------------------------

renderer.close()
gl_context.free()