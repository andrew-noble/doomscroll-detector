import cv2
from ultralytics import YOLO

# pose model
pmodel = YOLO("yolo11n-pose.pt")

# 0 = default webcam (use 1, 2, ... for other cameras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = pmodel(frame, verbose=False)[0]
    # print(results.keypoints.xy[0])

    for index, (x, y) in enumerate(results.keypoints.xy[0]):
        xc = int(x.item())
        yc = int(y.item())

        # cv2.circle(frame, (xc,yc), 5, (0, 0, 255), -1)

        if index in [5,6]: #shoulders
            cv2.circle(frame, (xc,yc), 10, (0, 0, 255), -1)
        elif index in [11, 12]:#hips
            cv2.circle(frame, (xc,yc), 10, (0, 255, 0), -1)
        elif index in [9, 10]: #wrists
            cv2.circle(frame, (xc,yc), 10, (255, 0, 0), -1)

    cv2.imshow("Webcam", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
