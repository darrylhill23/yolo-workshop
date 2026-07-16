import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# Load model
base_options = python.BaseOptions(
    model_asset_path="gesture-models/gesture_recognizer.task"
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2
)

recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = recognizer.recognize(mp_image)

    # Draw result
    if result.gestures:
        for g in result.gestures[0]:
            label = f"{g.category_name} ({g.score:.2f})"
            cv2.putText(frame, label, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for connection in mp_hands.HAND_CONNECTIONS:

                start = hand[connection.start]
                end = hand[connection.end]


                h, w, _ = frame.shape

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw points
            h, w, _ = frame.shape
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()