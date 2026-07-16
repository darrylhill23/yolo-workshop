import os
import re
import shutil

OLD = "dataset-old"
NEW = "dataset"

SPLITS = ["train", "val"]

img_exts = (".png", ".jpg", ".jpeg")

def extract_index(name):
    # expects frame_00012.png
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else -1


def get_max_index(folder):
    if not os.path.exists(folder):
        return 0

    max_idx = 0
    for f in os.listdir(folder):
        if f.lower().endswith(img_exts):
            idx = extract_index(f)
            max_idx = max(max_idx, idx)
    return max_idx


# ------------------------
# 1. find global max index in NEW dataset
# ------------------------
max_index = 0

for split in SPLITS:
    folder = os.path.join(NEW, "images", split)
    max_index = max(max_index, get_max_index(folder))

print(f"Starting new index from: {max_index + 1}")

# ------------------------
# 2. copy and rename old dataset
# ------------------------
current_index = max_index + 1

for split in SPLITS:
    old_img_dir = os.path.join(OLD, "images", split)
    old_lbl_dir = os.path.join(OLD, "labels", split)

    new_img_dir = os.path.join(NEW, "images", split)
    new_lbl_dir = os.path.join(NEW, "labels", split)

    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_lbl_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(old_img_dir)
        if f.lower().endswith(img_exts)
    ])

    for f in files:
        old_img_path = os.path.join(old_img_dir, f)

        base = os.path.splitext(f)[0]
        old_lbl_path = os.path.join(old_lbl_dir, base + ".txt")

        new_name = f"frame_{current_index:05d}"

        new_img_path = os.path.join(new_img_dir, new_name + os.path.splitext(f)[1])
        new_lbl_path = os.path.join(new_lbl_dir, new_name + ".txt")

        # copy image
        shutil.copy2(old_img_path, new_img_path)

        # copy label if exists
        if os.path.exists(old_lbl_path):
            shutil.copy2(old_lbl_path, new_lbl_path)
        else:
            # still create empty label file (valid YOLO negative sample)
            open(new_lbl_path, "w").close()

        current_index += 1

print("Done merging dataset-old into dataset.")