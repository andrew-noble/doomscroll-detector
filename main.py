import cv2
from ultralytics import YOLO

pmodel = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pmodel(frame, verbose=False)[0]
    kps = results.keypoints.xy  # shape (17,2)

    def pt(i):
        return tuple(map(int, kps[i].tolist()))

    if len(kps) > 0:
        kps = kps[0]
        # torso: shoulders → hips
        cv2.line(frame, pt(5), pt(6), (0,0,255), 2)   # shoulders
        cv2.line(frame, pt(11), pt(12), (0,255,0), 2) # hips
        cv2.line(frame, pt(5), pt(11), (255,255,0), 2)
        cv2.line(frame, pt(6), pt(12), (255,255,0), 2)

        # femurs: hip → knee
        cv2.line(frame, pt(11), pt(13), (255,0,0), 3) # left femur
        cv2.line(frame, pt(12), pt(14), (255,0,0), 3) # right femur
    else:
        pass

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
