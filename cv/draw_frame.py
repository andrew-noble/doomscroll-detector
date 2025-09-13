import cv2
import numpy as np

def draw_object_detection_frame(frame: np.ndarray, name: str, confidence: float, x1: int, y1: int, x2: int, y2: int):
    """
    Draw object detection bounding box and label on a frame.
    
    Args:
        frame (np.ndarray): The input frame to draw on
        name (str): The detected object class name
        confidence (float): Detection confidence score (0.0 to 1.0)
        x1 (int): Top-left x coordinate of bounding box
        y1 (int): Top-left y coordinate of bounding box
        x2 (int): Bottom-right x coordinate of bounding box
        y2 (int): Bottom-right y coordinate of bounding box
    
    Returns:
        np.ndarray: The frame with bounding box and label drawn
    """
    # Draw bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Prepare label text with confidence
    label = f"{name}: {confidence:.2f}"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Position label above the bounding box
    label_x = int(x1)
    label_y = int(y1) - 10
    if label_y < text_height:
        label_y = int(y1) + text_height + 10
    
    # Draw background rectangle for text
    cv2.rectangle(frame, 
                (label_x, label_y - text_height - baseline), 
                (label_x + text_width, label_y + baseline), 
                (0, 255, 0), -1)
    
    # Draw text
    cv2.putText(frame, label, (label_x, label_y), font, font_scale, (0, 0, 0), thickness)

    return frame


def draw_pose_frame(frame: np.ndarray, kps: np.ndarray):
    """
    Draw pose keypoints and skeleton on a frame.
    
    Args:
        frame (np.ndarray): The input frame to draw on
        kps (np.ndarray): Keypoints array with shape (17, 2) containing x,y coordinates
                          for each of the 17 COCO pose keypoints
    
    Returns:
        np.ndarray: The frame with pose skeleton drawn
    """
    def pt(i):
        return tuple(map(int, kps[i].tolist()))

    # Check if we have enough keypoints (need at least 15 for the skeleton)
    if len(kps) >= 15:
        # Draw keypoint dots
        # Red dots on shoulders (keypoints 5, 6)
        cv2.circle(frame, pt(5), 12, (0, 0, 255), -1)  # Left shoulder - red
        cv2.circle(frame, pt(6), 12, (0, 0, 255), -1)  # Right shoulder - red
        
        # Blue dots on hips (keypoints 11, 12)
        cv2.circle(frame, pt(11), 12, (255, 0, 0), -1)  # Left hip - blue
        cv2.circle(frame, pt(12), 12, (255, 0, 0), -1)  # Right hip - blue
        
        # Green dots on wrists (keypoints 9, 10)
        cv2.circle(frame, pt(9), 12, (0, 255, 0), -1)   # Left wrist - green
        cv2.circle(frame, pt(10), 12, (0, 255, 0), -1)  # Right wrist - green
        
        # OLD CODE - skeleton connections (commented out)
        # # torso: shoulders → hips
        # cv2.line(frame, pt(5), pt(6), (0,0,255), 2)   # shoulders
        # cv2.line(frame, pt(11), pt(12), (0,255,0), 2) # hips
        # cv2.line(frame, pt(5), pt(11), (255,255,0), 2)
        # cv2.line(frame, pt(6), pt(12), (255,255,0), 2)

        # # femurs: hip → knee
        # cv2.line(frame, pt(11), pt(13), (255,0,0), 3) # left femur
        # cv2.line(frame, pt(12), pt(14), (255,0,0), 3) # right femur
    else:
        # Not enough keypoints detected, skip drawing
        pass

    return frame