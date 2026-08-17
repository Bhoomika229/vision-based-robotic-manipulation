from pathlib import Path
import cv2


# ---------------------------------------------------------
# 1. Locate captured camera image
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "outputs" / "workspace_camera.png"


# ---------------------------------------------------------
# 2. Load image
# ---------------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Camera image loaded successfully.")

# Convert BGR -> HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# ---------------------------------------------------------
# 3. Detect red pixels
# ---------------------------------------------------------

# Red has two ranges in HSV
lower_red_1 = (0, 100, 100)
upper_red_1 = (10, 255, 255)

lower_red_2 = (170, 100, 100)
upper_red_2 = (180, 255, 255)

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


# ---------------------------------------------------------
# 4. Find contours
# ---------------------------------------------------------

contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# ---------------------------------------------------------
# 5. Find largest red object
# ---------------------------------------------------------

if not contours:
    print("No red object detected.")
    exit()

largest_contour = max(
    contours,
    key=cv2.contourArea
)

area = cv2.contourArea(largest_contour)

print(f"Detected red area: {area:.2f} pixels")


# Ignore tiny red noise
if area < 50:
    print("Red object is too small.")
    exit()


# ---------------------------------------------------------
# 6. Calculate bounding box
# ---------------------------------------------------------

x, y, w, h = cv2.boundingRect(
    largest_contour
)

center_x = x + w // 2
center_y = y + h // 2


# ---------------------------------------------------------
# 7. Print detected position
# ---------------------------------------------------------

print()
print("========== RED CUBE DETECTED ==========")
print(f"Bounding box:")
print(f"  x = {x}")
print(f"  y = {y}")
print(f"  width  = {w}")
print(f"  height = {h}")

print()
print("Cube image center:")
print(f"  X = {center_x} pixels")
print(f"  Y = {center_y} pixels")
print("========================================")


# ---------------------------------------------------------
# 8. Draw detection for visual verification
# ---------------------------------------------------------

output = image.copy()

cv2.rectangle(
    output,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

cv2.circle(
    output,
    (center_x, center_y),
    5,
    (255, 0, 0),
    -1
)

cv2.putText(
    output,
    f"Cube: ({center_x}, {center_y})",
    (x, max(y - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 0),
    2
)


# ---------------------------------------------------------
# 9. Save detection result
# ---------------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "detected_cube.png"
)

cv2.imwrite(
    str(output_path),
    output
)

print()
print(f"Detection image saved to:")
print(output_path)