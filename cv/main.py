import cv2
import signal
import sys
from draw_frame import draw_status_overlay, draw_pose_frame
from vision import detect_doomscrolling, detect_holding_phone, detect_reclined, get_pose, get_phones
from opts import get_opts
from collections import deque
import time
import requests

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

# post requests setup
url = "http://localhost:8000/data"

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

    WINDOW_SEC = 5.0
    ENTER, EXIT = 0.60, 0.30 # to enter doomscroll state, pos > 60%, exit: pos < 30%
    buf = deque()
    pos_count = 0
    state = False           # buffered on/off
    last_post = 0.0         # last POST time (epoch seconds)

    def update(t, is_pos, window_sec=WINDOW_SEC, enter=ENTER, exit=EXIT):
        nonlocal pos_count, state
        # push new
        buf.append((t, is_pos))
        pos_count += int(is_pos)

        # drop old
        cutoff = t - window_sec
        while buf and buf[0][0] < cutoff:
            _, old = buf.popleft()
            pos_count -= int(old)

        if not buf:
            return state

        frac = pos_count / len(buf)
        if not state and frac >= enter:
            state = True
        elif state and frac <= exit:
            state = False
        return state

    global running
    while running:
        t = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # run models
        frame, kps_normalized, kps = get_pose(frame)
        frame, phones_normalized = get_phones(frame) #phones boxes drawn in here, would need fixing.

        # heuristics
        is_reclining = detect_reclined(frame, kps_normalized, threshold=opts.reclined_threshold)
        is_holding_phone = detect_holding_phone(frame, phones_normalized, kps_normalized, threshold=opts.holding_phone_threshold)
        is_doomscrolling = is_reclining and is_holding_phone

        # sliding-window + hysteresis
        is_buffered = update(t, is_doomscrolling)

        # draw overlays
        # frame = draw_status_overlay(frame, is_doomscrolling, "Doomscrolling (raw)", "top_left")
        # phone boxes are drawn in the get_phones func, very awkward, but okay since last CV step
        frame = draw_pose_frame(frame, kps)
        frame = draw_status_overlay(frame, is_buffered, "Doomscrolling", "bottom_left")

        # periodic POST every ~5s
        if t - last_post >= 1.0:
            try:
                requests.post(
                    "http://localhost:8000/api/data",
                    json={"doomscrolling": bool(is_buffered), "timestamp": t},
                    timeout=1.5
                )
            except requests.RequestException as e:
                print("POST error:", e)
            last_post = t

        if opts.record_video:
            out.write(frame)

        display_frame = cv2.resize(frame, (1280, 720))
        if not opts.headless:
            cv2.imshow("Webcam", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if opts.record_video:
        print("Video saved as 'doomscroll_recording.mp4'")

if __name__ == "__main__":
    main()

