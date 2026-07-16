import socket
import struct
import threading
import cv2
import numpy as np
import json
import time

from collections import deque
from ultralytics import YOLO

# model = YOLO("../yolov8n.pt")

model = YOLO("../runs/train/exp/weights/best.pt")

# comment out if using cpu-only
model.to("cuda")


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


            results = model(frame, verbose= False, conf=0.05)[0]
            annotated = results.plot()


            # --- FPS calculation ---
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time

            fps = 1.0 / dt if dt > 0 else 0
            fps_history.append(fps)

            smooth_fps = sum(fps_history) / len(fps_history)

            
            cv2.putText(
                annotated,
                f"FPS: {smooth_fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            latest_frame = annotated  # shared buffer

            # response
            boxes = results.boxes

            det_list = []

            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                det_list.append({
                    "label": model.names[cls],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

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