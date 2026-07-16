import cv2
import os

img_dir = "candidates"
out_img_dir = "dataset/images/train"
out_lbl_dir = "dataset/labels/train"

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

drawing = False
box = None
x1 = y1 = x2 = y2 = 0

def mouse(event, x, y, flags, param):
    global drawing, x1, y1, x2, y2, box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = x, y
        box = (x1, y1, x2, y2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        box = (x1, y1, x2, y2)

cv2.namedWindow("img")
cv2.setMouseCallback("img", mouse)

for fname in os.listdir(img_dir):
    path = os.path.join(img_dir, fname)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    box = None
    label_lines = None

    while True:
        disp = img.copy()

        if box:
            cv2.rectangle(disp, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)

        cv2.imshow("img", disp)
        key = cv2.waitKey(1)

        # negative sample
        if key == ord('n'):
            label_lines = ""
            break

        # positive sample
        if key == 13 and box:  # ENTER
            x_c = ((box[0] + box[2]) / 2) / w
            y_c = ((box[1] + box[3]) / 2) / h
            bw = abs(box[2] - box[0]) / w
            bh = abs(box[3] - box[1]) / h

            label_lines = f"0 {x_c} {y_c} {bw} {bh}"
            break

    # save files
    img_out = os.path.join(out_img_dir, fname)
    lbl_out = os.path.join(out_lbl_dir, fname.replace(".png", ".txt"))

    cv2.imwrite(img_out, img)

    with open(lbl_out, "w") as f:
        if label_lines:
            f.write(label_lines)
        else:
            f.write("")  # negative image

cv2.destroyAllWindows()