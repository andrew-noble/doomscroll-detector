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
