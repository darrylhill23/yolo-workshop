import os
import random
import shutil

img_dir = "dataset/images/train"
lbl_dir = "dataset/labels/train"

out_img_val = "dataset/images/val"
out_lbl_val = "dataset/labels/val"

os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)
os.makedirs(out_img_val, exist_ok=True)
os.makedirs(out_lbl_val, exist_ok=True)

# Move existing validation data back into training
for src_dir, dst_dir in [
    (out_img_val, img_dir),
    (out_lbl_val, lbl_dir)
]:
    for f in os.listdir(src_dir):
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)

        if os.path.isfile(src):
            shutil.move(src, dst)

images = [f for f in os.listdir(img_dir) if f.endswith(".png")]

empty = []
non_empty = []

# classify labels
for img in images:
    label_path = os.path.join(lbl_dir, img.replace(".png", ".txt"))

    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        empty.append(img)
    else:
        non_empty.append(img)

# sample 20% from each group
val_empty = random.sample(empty, max(1, int(0.2 * len(empty))))
val_non_empty = random.sample(non_empty, max(1, int(0.2 * len(non_empty))))

val_set = set(val_empty + val_non_empty)

# move
for img in val_set:
    img_src = os.path.join(img_dir, img)
    lbl_src = os.path.join(lbl_dir, img.replace(".png", ".txt"))

    shutil.move(img_src, os.path.join(out_img_val, img))

    if os.path.exists(lbl_src):
        shutil.move(lbl_src, os.path.join(out_lbl_val, img.replace(".png", ".txt")))
    else:
        open(os.path.join(out_lbl_val, img.replace(".png", ".txt")), "w").close()
