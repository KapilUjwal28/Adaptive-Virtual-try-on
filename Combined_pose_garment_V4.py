import cv2
import numpy as np
import mediapipe as mp

# Initialize pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def get_landmarks(image_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks:
        print("No landmarks found.")
        return None, image

    lm = results.pose_landmarks.landmark
    h, w = image.shape[:2]

    def pt(i): return int(lm[i].x * w), int(lm[i].y * h)

    return {
        "l_shoulder": pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
        "r_shoulder": pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
        "l_elbow": pt(mp_pose.PoseLandmark.LEFT_ELBOW),
        "r_elbow": pt(mp_pose.PoseLandmark.RIGHT_ELBOW),
        "l_hip": pt(mp_pose.PoseLandmark.LEFT_HIP),
        "r_hip": pt(mp_pose.PoseLandmark.RIGHT_HIP)
    }, image

# Paths
person_img_path = "C:/Users/kapil/Btech/Sem 4/SoftComputing/photos/half_man_photo.jpg"
tshirt_img_path = "yellow_tshirt_photo.jpg"

# Get pose
pose_pts, person_img = get_landmarks(person_img_path)
if pose_pts is None:
    exit()

# Load T-shirt
tshirt = cv2.imread(tshirt_img_path)
if tshirt is None:
    print("T-shirt image missing.")
    exit()

# Resize for mapping
src_h, src_w = tshirt.shape[:2]
offset_x = 50  # widen t-shirt shoulders
offset_y = 20  # raise t-shirt slightly upwards

src_pts = np.float32([
    [27, 57],
    [171, 54],
    [48, 230],
    [157, 229]     # right bottom of t-shirt
])

# Destination points from person image pose
dst_pts = np.float32([
    pose_pts["l_shoulder"],
    pose_pts["r_shoulder"],
    pose_pts["l_hip"],
    pose_pts["r_hip"]
])

# Perspective transform
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(tshirt, matrix, (person_img.shape[1], person_img.shape[0]))

# Create mask
gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Apply to person
inv_mask = cv2.bitwise_not(mask)
bg = cv2.bitwise_and(person_img, person_img, mask=inv_mask)
fg = cv2.bitwise_and(warped, warped, mask=mask)
final = cv2.add(bg, fg)
# cv2.imwrite("final_overlay_output.jpg", final)
cv2.imshow("Final Overlay Result", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
