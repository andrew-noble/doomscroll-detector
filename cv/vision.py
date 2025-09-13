from ultralytics import YOLO
import numpy as np
from draw_frame import draw_object_detection_frame
from draw_frame import draw_pose_frame

pose_model = YOLO("yolo11n-pose.pt")
detection_model = YOLO("yolo11n.pt")

def get_phones(frame: np.ndarray, draw: bool = True) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    detection_results = detection_model(frame, verbose=False)[0]
    boxes = detection_results.boxes

    # Filter for cell phone detections - much simpler!
    phones = boxes[boxes.cls == 67]  # 67 is the COCO class ID for cell phone

    phone_coords_normalized = []
    for box in phones:
        x1_n, y1_n, x2_n, y2_n = box.xyxyn.cpu().numpy()[0]
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        phone_coords_normalized.append((x1_n, y1_n, x2_n, y2_n))

        if draw:
            name = "cell phone"
            confidence = float(box.conf[0])
            frame = draw_object_detection_frame(frame, name, confidence, x1, y1, x2, y2)
    
    return frame, phone_coords_normalized

def get_pose(frame: np.ndarray, draw: bool = True) -> tuple[np.ndarray, np.ndarray]:

    # currently we only get one person
    results = pose_model(frame, verbose=False)[0]
    
    # Check if any keypoints were detected
    if len(results.keypoints) > 0:
        kps_normalized = results.keypoints.xyn[0].cpu().numpy()  # shape (17,2)
        kps = results.keypoints.xy[0].cpu().numpy()  # shape (17,2)
        
        # Debug: print keypoint info
        # print(f"Keypoints shape: {kps.shape}, Non-zero keypoints: {np.sum(kps > 0)}")
        
        if draw:
            frame = draw_pose_frame(frame, kps)
    else:
        # No person detected
        print("No person detected")
        kps_normalized = np.zeros((17, 2))
        kps = np.zeros((17, 2))

    return frame, kps_normalized

def detect_reclined(frame: np.ndarray, kps_normalized: np.ndarray, draw: bool = True) -> bool:
    margin = 0.2
    
    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    y_shoulder = midpoint(kps_normalized[5], kps_normalized[6])[1]
    y_hips = midpoint(kps_normalized[11], kps_normalized[12])[1]
    y_knee = midpoint(kps_normalized[13], kps_normalized[14])[1]

    vert_diff = abs(y_hips - y_shoulder)

    is_reclining = vert_diff < margin

    return is_reclining

def detect_holding_phone(frame: np.ndarray, phones_n: list[tuple[int, int, int, int]], kps_n: np.ndarray, draw: bool = True) -> bool:
    radius = 0.5 # will need tweaking

    if len(phones_n) == 0:
        return False

    def box_center_xyxy(box):
        x1, y1, x2, y2 = map(float, box)  # handles np.float32 scalars too
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    for phone in phones_n:
        phone_coords = np.array(phone)
        phone_midpoint = box_center_xyxy(phone_coords) # gets center of phone box

        dist_left = np.linalg.norm(phone_midpoint - kps_n[9])
        dist_right = np.linalg.norm(phone_midpoint - kps_n[10])

        if dist_left < radius or dist_right < radius:
            return True

    return False

def detect_doomscrolling(frame: np.ndarray, kps: np.ndarray, draw: bool = True):
    frame, phones = get_phones(frame, draw)
    frame, kps = get_pose(frame, draw)

    is_reclining = detect_reclined(frame, kps, draw)
    is_holding_phone = detect_holding_phone(frame, phones, kps, draw)

    return is_reclining, is_holding_phone