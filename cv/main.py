import cv2
import signal
import sys
from draw_frame import draw_status_overlay
from vision import detect_doomscrolling, detect_holding_phone, detect_reclined, get_pose, get_phones
from opts import get_opts


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

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter('doomscroll_recording.mp4', fourcc, 30.0, (int(actual_width), int(actual_height)))


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
    opts = get_opts()

    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        # run models
        frame, kps_normalized = get_pose(frame, draw=not opts.headless)
        frame, phones_normalized = get_phones(frame, draw=not opts.headless)

        # run doomscrolling heuristics
        is_reclining = detect_reclined(frame, kps_normalized, threshold=opts.reclined_threshold)
        is_holding_phone = detect_holding_phone(frame, phones_normalized, kps_normalized, threshold=opts.holding_phone_threshold)

        is_doomscrolling = is_reclining and is_holding_phone

        frame = draw_status_overlay(frame, is_doomscrolling, "Doomscrolling", "top_left")

        # Write frame to video file
        if opts.record_video:
            out.write(frame)
        
        # Resize frame for larger display (optional)
        display_frame = cv2.resize(frame, (1280, 720))  # Make display larger
        cv2.imshow("Webcam", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    out.release()  # Release video writer
    cv2.destroyAllWindows()
    if opts.record_video:
        print("Video saved as 'doomscroll_recording.mp4'")

if __name__ == "__main__":
    main()

