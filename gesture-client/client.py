import socket
import struct
import threading
import cv2
import numpy as np
import json
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

from collections import deque
from ultralytics import YOLO

# Load model
base_options = python.BaseOptions(
    model_asset_path="../gesture-models/gesture_recognizer.task"
)

options = vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2
)

recognizer = vision.GestureRecognizer.create_from_options(options)


SERVER_IP = "127.0.0.1"
#SERVER_IP = "172.17.22.188"
PORT = 5001

# robot camera resolution
#HEIGHT = 240
#WIDTH = 320

# localhost camera resolution
HEIGHT = 480
WIDTH = 640

latest_frame = None
running = True


def recvall(conn, size):
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data




def handle_server(conn):
    global latest_frame
    prev_time = time.time()
    fps_history = deque(maxlen=10)
    # fps = 0
    try:
        while True:
            header = recvall(conn, 4)
            if not header:
                break

            (size,) = struct.unpack("!I", header)
            img_bytes = recvall(conn, size)
            if img_bytes is None:
                break

            frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            results = recognizer.recognize(mp_image)

            # --- FPS calculation ---
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(fps)

            smooth_fps = sum(fps_history) / len(fps_history)

            cv2.putText(
                frame,
                f"FPS: {smooth_fps:.2f}",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            

            # response
            

            det_list = []
            if results.gestures:
                for g in  results.gestures[0]:
                    
                    label = f"{g.category_name} ({g.score:.2f})"
                    cv2.putText(frame, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                    det_list.append({
                        "name": g.category_name,
                        "confidence": g.score,
                    })

            latest_frame = frame  # shared buffer
            resp = json.dumps(det_list).encode()

            conn.sendall(struct.pack("!I", len(resp)))
            conn.sendall(resp)
            #print("send", time.time())

    finally:
        conn.close()


print("Connecting to server...")

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER_IP, PORT))

print("Client connected...")



# start networking thread
threading.Thread(target=handle_server, args=(client,), daemon=True).start()


# ---------------- GUI LOOP (IMPORTANT PART) ----------------
cv2.namedWindow("Client", cv2.WINDOW_NORMAL)

while True:
    if latest_frame is not None:
        cv2.imshow("Client", latest_frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()