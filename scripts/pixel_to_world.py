from pathlib import Path
import cv2
import numpy as np


# =========================================================
# 1. Project settings
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "outputs" / "workspace_camera.png"


# =========================================================
# 2. Camera parameters
# =========================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

CAMERA_HEIGHT = 4.5

# Camera field of view
FOV_DEGREES = 90.0


# =========================================================
# 3. Load camera image
# =========================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Camera image loaded.")


# =========================================================
# 4. Detect red cube
# =========================================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red_1 = np.array([0, 100, 100])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([170, 100, 100])
upper_red_2 = np.array([180, 255, 255])

mask1 = cv2.inRange(
    hsv,
    lower_red_1,
    upper_red_1
)

mask2 = cv2.inRange(
    hsv,
    lower_red_2,
    upper_red_2
)

mask = mask1 | mask2


# =========================================================
# 5. Find cube contour
# =========================================================

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No red object detected.")


largest_contour = max(
    contours,
    key=cv2.contourArea
)

area = cv2.contourArea(largest_contour)

if area < 50:
    raise RuntimeError(
        "Detected red object is too small."
    )


# =========================================================
# 6. Calculate image coordinates
# =========================================================

x, y, w, h = cv2.boundingRect(
    largest_contour
)

pixel_x = x + w / 2
pixel_y = y + h / 2

print()
print("========== VISION RESULT ==========")
print(f"Cube pixel X: {pixel_x:.2f}")
print(f"Cube pixel Y: {pixel_y:.2f}")
print(f"Cube area: {area:.2f} pixels")


# =========================================================
# 7. Calculate camera focal length
# =========================================================

fov_rad = np.deg2rad(FOV_DEGREES)

focal_length = (
    IMAGE_HEIGHT / 2
) / np.tan(fov_rad / 2)

print()
print(f"Camera focal length: {focal_length:.2f} pixels")


# =========================================================
# 8. Convert pixel coordinates to world coordinates
# =========================================================

# Principal point
cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2


# Horizontal displacement from image center
dx = pixel_x - cx

# Vertical displacement from image center
dy = pixel_y - cy


# Since the camera is directly above the workspace,
# the ground-plane coordinates can be calculated from
# the camera ray.

world_x = (dx / focal_length) * CAMERA_HEIGHT

world_y = -(dy / focal_length) * CAMERA_HEIGHT


# The cube is sitting above the ground.
# We use the workspace/cube height here temporarily.
world_z = 0.25


# =========================================================
# 9. Print estimated 3D position
# =========================================================

print()
print("========== WORLD POSITION ==========")

print(f"Estimated X: {world_x:.3f} m")
print(f"Estimated Y: {world_y:.3f} m")
print(f"Estimated Z: {world_z:.3f} m")

print("====================================")