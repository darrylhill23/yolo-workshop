import cv2
import os

def parse_time(t):
    parts = list(map(int, t.split(":")))
    if len(parts) == 2:       # MM:SS
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3:     # HH:MM:SS
        h, m, s = parts
        return h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {t}")

def extract_timeframes(video_path, out_dir, ranges, step_s):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    idx = 0

    for start_t, end_t in ranges:
        start_s = parse_time(start_t)
        end_s = parse_time(end_t)

        t = start_s
        while t <= end_s:
            frame_no = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.png"), frame)

            idx += 1
            t += step_s

    cap.release()

# You can adjust this if you only want to use frames from a certain time range. Currently set for the first hour of the video.
ranges = [
    ("00:00", "1:00:00"),
]

# chopping output.mp4 into frames and saving them in the candidates folder, 1 frame per second
extract_timeframes("output.mp4", "candidates", ranges, step_s=1)
