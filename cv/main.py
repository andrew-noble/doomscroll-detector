import cv2
import signal
import sys
from draw_frame import draw_object_detection_frame
from vision import detect_doomscrolling, detect_reclined, get_pose

cap = cv2.VideoCapture(2)

# Set higher resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

# Optional: Set FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if resolution was set successfully
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

# Global flag for clean shutdown
running = True

def signal_handler(sig, frame):
    global running
    print('\nShutting down gracefully...')
    running = False
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # isReclined, isHoldingPhone = detect_doomscrolling(frame)

        
        # if isReclined and isHoldingPhone:
        #     print("Doomscrolling detected")
        # else:
        #     print("No doomscrolling detected")

        frame, kps_normalized = get_pose(frame)
        is_reclining = detect_reclined(frame, kps_normalized)
        print(is_reclining)

        # Resize frame for larger display (optional)
        display_frame = cv2.resize(frame, (1280, 720))  # Make display larger
        cv2.imshow("Webcam", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

