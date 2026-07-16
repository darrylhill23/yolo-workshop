import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow("recording", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
out.release()
cv2.destroyAllWindows()