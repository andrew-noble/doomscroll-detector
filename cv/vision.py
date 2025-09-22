from ultralytics import YOLO
import numpy as np

pose_model = YOLO("yolo11n-pose.pt")
detection_model = YOLO("yolo11n.pt")

def get_phone(frame: np.ndarray) -> tuple[np.ndarray, tuple[np.float32, np.float32, np.float32, np.float32] | None, tuple[int, int, int, int] | None]:
    detection_results = detection_model(frame, conf=0.1, verbose=False)[0] # default confidence is 0.5! Definitely want more sensitivity
    boxes = detection_results.boxes

    # Filter for cell phone detections - much simpler!
    phones = boxes[boxes.cls == 67]  # 67 is the COCO class ID for cell phone

    if len(phones) == 0:
        return frame, None, None

    box = phones[0]  # this project isn't robust to multiple phones!

    x1_n, y1_n, x2_n, y2_n = box.xyxyn.cpu().numpy()[0] # packaged in an an extra array for some reason
    phone_coords_normalized = (x1_n, y1_n, x2_n, y2_n)

    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
    phone_coords = (x1, y1, x2, y2)
    
    return frame, phone_coords_normalized, phone_coords

def get_pose(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # currently we only get one person
    results = pose_model(frame, verbose=False)[0]
    
    # Check if any keypoints were detected
    if len(results.keypoints) > 0:
        kps_normalized = results.keypoints.xyn[0].cpu().numpy()  # shape (17,2)
        kps = results.keypoints.xy[0].cpu().numpy()  # shape (17,2)
        
        # Debug: print keypoint info
        # print(f"Keypoints shape: {kps.shape}, Non-zero keypoints: {np.sum(kps > 0)}")

    else:
        # No person detected
        print("No person detected")
        kps_normalized = np.zeros((17, 2))
        kps = np.zeros((17, 2))

    return frame, kps_normalized, kps

def detect_reclined(frame: np.ndarray, kps_normalized: np.ndarray, threshold: float) -> bool:
    
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    y_shoulder = midpoint(kps_normalized[5], kps_normalized[6])[1]
    y_hips = midpoint(kps_normalized[11], kps_normalized[12])[1]
    y_knee = midpoint(kps_normalized[13], kps_normalized[14])[1]

    vert_diff = abs(y_hips - y_shoulder)

    is_reclining = vert_diff < threshold

    return is_reclining

def detect_holding_phone(frame: np.ndarray, phone_n: tuple[int, int, int, int] | None, kps_n: np.ndarray, threshold: float, *, draw: bool = True) -> bool:

    if phone_n is None: # no phone in frame
        return False

    def box_center_xyxy(box):
        x1, y1, x2, y2 = map(float, box)  # handles np.float32 scalars too
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    phone_coords = np.array(phone_n)
    phone_midpoint = box_center_xyxy(phone_coords) # gets center of phone box

    dist_left = np.linalg.norm(phone_midpoint - kps_n[9])
    dist_right = np.linalg.norm(phone_midpoint - kps_n[10])

    if dist_left < threshold or dist_right < threshold:
        return True
