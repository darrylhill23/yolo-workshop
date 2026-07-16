import socket
import struct
import cv2
import time
import json

HOST ="0.0.0.0" 
PORT = 5001


def receiveAll(sock, size):
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

'''
The change is, I still set up to read the frames of the video.
Anyone who connects to me, I send them frames and read the response. 
I only need one connection. 
'''


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen()

print("Server listening...")





# start networking thread
# threading.Thread(target=server_thread, daemon=True).start()

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client.connect((SERVER_IP, PORT))

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width, height)

client, addr = server.accept()
print("Client:", addr)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = frame.tobytes()

    client.sendall(struct.pack("!I", len(data)))
    client.sendall(data)

    # receive response
    header = receiveAll(client, 4)
    print("recv", time.time())
    if not header:
        break

    (size,) = struct.unpack("!I", header)
    resp = receiveAll(client, size)
    # resp = receiveAll(client, size)
    data = json.loads(resp.decode())

    if len(data) == 0:
        print("No detections")
    else:
        print("Detections:", data)

   
    

client.close()