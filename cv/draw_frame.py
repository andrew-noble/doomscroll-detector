import cv2
import numpy as np

def draw_object_detection_frame(frame: np.ndarray, coords: tuple[np.float32, np.float32, np.float32, np.float32] | None, color: str = "green"):
    """
    Draw object detection bounding box and label on a frame.
    
    Args:
        frame (np.ndarray): The input frame to draw on
        coords: Bounding box coordinates (x1, y1, x2, y2)
        color (str): Color name ("green", "red", "blue", "yellow", "purple", "orange", "cyan", "white")
    
    Returns:
        np.ndarray: The frame with bounding box and label drawn
    """
    
    # Color mapping (BGR format)
    colors = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "orange": (0, 165, 255),
        "cyan": (255, 255, 0),
        "white": (255, 255, 255)
    }
    bgr_color = colors.get(color.lower(), (0, 255, 0))  # Default to green
    
    if coords:
        x1, y1, x2, y2 = coords
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), bgr_color, 2)        
        # Prepare label text
        label = "cellphone"
        
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
                    bgr_color, -1)
        
        # Draw text
        cv2.putText(frame, label, (label_x, label_y), font, font_scale, (0, 0, 0), thickness)

    else:
        pass
        # print("No coords provided, skipping drawing")

    # print(f"draw_object_detection_frame returning frame shape: {frame.shape}")
    return frame


def draw_pose_frame(frame: np.ndarray, kps: np.ndarray, color: str = "green"):
    """
    Draw pose keypoints and skeleton on a frame.
    
    Args:
        frame (np.ndarray): The input frame to draw on
        kps (np.ndarray): Keypoints array with shape (17, 2) containing x,y coordinates
                          for each of the 17 COCO pose keypoints
        color (str): Color name ("green", "red", "blue", "yellow", "purple", "orange", "cyan", "white")
    
    Returns:
        np.ndarray: The frame with pose skeleton drawn
    """
    # Color mapping (BGR format)
    colors = {
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "orange": (0, 165, 255),
        "cyan": (255, 255, 0),
        "white": (255, 255, 255)
    }
    bgr_color = colors.get(color.lower(), (0, 255, 0))  # Default to green
    
    def pt(i):
        return tuple(map(int, kps[i].tolist()))

    # Check if we have enough keypoints (need at least 15 for the skeleton)
    if len(kps) >= 15:
        # Draw keypoint dots with the specified color
        frame = cv2.circle(frame, pt(5), 30, bgr_color, -1)   # Left shoulder
        frame = cv2.circle(frame, pt(6), 30, bgr_color, -1)   # Right shoulder
        frame = cv2.circle(frame, pt(11), 30, bgr_color, -1)  # Left hip
        frame = cv2.circle(frame, pt(12), 30, bgr_color, -1)  # Right hip
        frame = cv2.circle(frame, pt(9), 30, bgr_color, -1)   # Left wrist
        frame = cv2.circle(frame, pt(10), 30, bgr_color, -1)  # Right wrist
        
        # OLD CODE - skeleton connections
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


def draw_status_overlay(frame: np.ndarray, status: bool, label: str = "Status", position: str = "top_left"):
    """
    Draw a simple boolean status indicator in the corner of the frame.
    
    Args:
        frame (np.ndarray): The input frame to draw on
        status (bool): The boolean status to display
        label (str): Label text to show next to the indicator
        position (str): Position of the overlay ("top_left", "top_right", "bottom_left", "bottom_right")
    
    Returns:
        np.ndarray: The frame with status overlay drawn
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Set colors - Green for False, Red for True (inverted logic)
    text_color = (0, 255, 0) if not status else (0, 0, 255)  # Green for False, Red for True
    
    # Create status text
    status_text = f"{label}: {'TRUE' if status else 'FALSE'}"
    
    # Set font properties - much bigger
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Much bigger text
    thickness = 4  # Thicker text
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, thickness)
    
    # Set padding - much bigger
    padding = 20
    
    # Calculate position based on corner
    if position == "top_left":
        x = padding
        y = text_height + padding
    elif position == "top_right":
        x = width - text_width - padding
        y = text_height + padding
    elif position == "bottom_left":
        x = padding
        y = height - padding
    elif position == "bottom_right":
        x = width - text_width - padding
        y = height - padding
    else:
        x = padding
        y = text_height + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, 
                  (x - padding//2, y - text_height - padding//2), 
                  (x + text_width + padding//2, y + baseline + padding//2), 
                  (0, 0, 0), -1)  # Black background
        
    # Draw text
    cv2.putText(frame, status_text, (x, y), font, font_scale, text_color, thickness)
    
    return frame

def tint_red(frame):

    overlay = frame.copy()
    red = (0, 0, 255)   # BGR
    cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), red, -1)
    alpha = 0.2         # 0=transparent, 1=solid red
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame
