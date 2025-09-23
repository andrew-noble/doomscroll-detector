import cv2
import signal
import sys
from draw_frame import draw_object_detection_frame, draw_status_overlay, draw_pose_frame, draw_text_overlay, tint_red
from vision import get_pose, get_phone
from heuristics import check_holding_phone, check_reclined
from opts import get_opts
from collections import deque
import time

cap = cv2.VideoCapture(0)

# Set higher resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)   # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Height

# Check if resolution was set successfully
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")

# Video recording setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter('doomscroll_recording.mp4', fourcc, 10, (int(actual_width), int(actual_height))) # measured ~5 fps on my machine

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

    WINDOW_SEC = 1.0
    ENTER, EXIT = 0.60, 0.30 # to enter doomscroll state, pos > 60%, exit: pos < 30%
    buf = deque()
    pos_count = 0
    state = False           # buffered on/off

    def update_buffer(t, is_pos, window_sec=WINDOW_SEC, enter=ENTER, exit=EXIT):
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

    penalty = 0
    penalty_rate = 0.15 # spoof charge rate per second in USD
    last = time.time()

    frame_count = 0
    start_time = time.time()

    global running
    while running:
        t = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # run models
        frame, kps_normalized, kps = get_pose(frame)
        frame, phone_coords_normalized, phone_coords = get_phone(frame) 
        
        # heuristics
        is_reclining = check_reclined(frame, kps_normalized, threshold=opts.reclined_threshold)
        is_scrolling = check_holding_phone(frame, phone_coords_normalized, kps_normalized, threshold=opts.holding_phone_threshold)
        print(f'Phone: {is_scrolling}, Pose: {is_reclining}')
        is_doomscrolling = is_reclining and is_scrolling

        # sliding-window + hysteresis
        is_buffered = update_buffer(t, is_doomscrolling)

        color = "red" if is_buffered else "green"

        # draw overlays
        frame = draw_pose_frame(frame, kps, color, wrist_bound=opts.holding_phone_threshold)
        frame = draw_object_detection_frame(frame, phone_coords, color)
        frame = draw_status_overlay(frame, is_buffered, "Doomscrolling", "top_left")

        # update penalty every buffered window. I think this is not ideal with the buffering, doesn't cleanly track stackup, but good enough for now. 
        now = time.time()
        if now - last >= WINDOW_SEC and is_buffered:
            penalty += penalty_rate
            last = now

        frame = draw_text_overlay(frame, f'Penalty: ${penalty:.2f}', "top_right")

        if is_buffered:
            frame = tint_red(frame)

        frame_count += 1

        if frame_count >= 60:  # check after ~60 frames
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Approx FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()

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

