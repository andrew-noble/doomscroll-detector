import numpy as np

def check_reclined(frame: np.ndarray, kps_normalized: np.ndarray, threshold: float) -> bool:
    """
    Works by checking the vertical distance between shoulderline and hipline. 
    """

    def midpoint(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    y_shoulder = midpoint(kps_normalized[5], kps_normalized[6])[1]
    y_hips = midpoint(kps_normalized[11], kps_normalized[12])[1]
    y_knee = midpoint(kps_normalized[13], kps_normalized[14])[1]

    vert_diff = abs(y_hips - y_shoulder)

    # print(f'y_shoulder: {y_shoulder}, y_hips: {y_hips}, diff: {vert_diff}')

    is_reclining = vert_diff < threshold

    return is_reclining

def check_holding_phone(frame: np.ndarray, phone_n: tuple[int, int, int, int] | None, kps_n: np.ndarray, threshold: float) -> bool:
    """
    Currently this heuristic just works off a radius in 2D from the wrist joint to the center of the detected phone. Pretty simple, but works. 
    """

    if phone_n is None: # no phone in frame
        return False

    def box_center_xyxy(box): # gets center of phone in normalized coords
        x1, y1, x2, y2 = map(float, box)  # handles np.float32 scalars too
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    phone_coords = np.array(phone_n)
    phone_midpoint = box_center_xyxy(phone_coords)

    dist_left = np.linalg.norm(phone_midpoint - kps_n[9])
    dist_right = np.linalg.norm(phone_midpoint - kps_n[10])

    if dist_left < threshold or dist_right < threshold:
        return True

    else: # phone in frame, but not near hands
        return False