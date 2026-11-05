#!/usr/bin/env python3
"""UFO Attack: dodge alien projectiles with a YOLO-detected real object.

Dependencies: pip install ultralytics opencv-python numpy
Run: python ufo_attack.py --model runs/train/exp/weights/best.pt
"""
import argparse
import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class UFO:
    x: float
    y: float
    phase: float
    cooldown: float
    orbit_radius: float
    orbit_speed: float
    drift_x: float = 0.0
    drift_y: float = 0.0
    wander_timer: float = 0.0


@dataclass
class Shot:
    x: float
    y: float
    vx: float
    vy: float
    ttl: float = 3.0


@dataclass
class Blast:
    x: float
    y: float
    age: float = 0.0


def spawn_ufo(w, h):
    return UFO(random.uniform(30, w - 30), random.uniform(25, h * 0.35),
               random.uniform(0, 6.28), random.uniform(0.8, 2.0),
               random.uniform(85, 190), random.choice((-1, 1)) * random.uniform(0.35, 0.9))


def draw_ufo(frame, ufo, t):
    x, y = int(ufo.x), int(ufo.y + 4 * math.sin(t * 3 + ufo.phase))
    cv2.ellipse(frame, (x, y), (27, 10), 0, 0, 360, (185, 190, 205), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y - 7), (14, 12), 0, 180, 360, (235, 180, 70), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (x, y), (27, 10), 0, 0, 360, (80, 90, 110), 2, cv2.LINE_AA)
    for dx in (-16, 0, 16):
        cv2.circle(frame, (x + dx, y + 3), 3, (0, 245, 255), -1, cv2.LINE_AA)


def draw_hud(frame, hp, score, status):
    cv2.rectangle(frame, (8, 8), (310, 80), (15, 15, 25), -1)
    cv2.putText(frame, f'HITS: {score}', (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (18, 43), (218, 60), (70, 70, 70), -1)
    cv2.rectangle(frame, (18, 43), (18 + max(0, hp) * 2, 60),
                  (40, 200, 70) if hp > 30 else (40, 40, 230), -1)
    cv2.putText(frame, status, (225, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Move the object to dodge!  R: restart  Q: quit',
                (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/train/exp/weights/best.pt')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=160)
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--ufos', type=int, default=3)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera {args.camera}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError('Could not read webcam frame')
    h, w = first.shape[:2]

    def reset():
        return [spawn_ufo(w, h) for _ in range(args.ufos)], [], [], 100, 0, 0.0

    ufos, shots, blasts, hp, score, invulnerable = reset()
    target = None
    box = None
    last_seen = -100.0
    last = time.monotonic()
    start = last

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            elapsed = now - start

            if hp > 0:
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                       max_det=1, verbose=False)[0]
                if len(result.boxes):
                    coords = result.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, coords)
                    detected = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
                    target = detected if target is None else 0.65 * target + 0.35 * detected
                    box = (x1, y1, x2, y2)
                    last_seen = now
                elif now - last_seen > 0.4:
                    target = None
                    box = None

                if target is not None:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, tuple(target.astype(int)), 7, (0, 255, 255), 2)

                for u in ufos:
                    if target is not None:
                        # Each UFO orbits at its own radius, direction and phase.
                        angle = elapsed * u.orbit_speed + u.phase
                        u.wander_timer -= dt
                        if u.wander_timer <= 0:
                            u.drift_x = random.uniform(-45, 45)
                            u.drift_y = random.uniform(-25, 25)
                            u.wander_timer = random.uniform(1.0, 3.0)
                        desired_x = target[0] + math.cos(angle) * u.orbit_radius + u.drift_x
                        desired_y = target[1] - 90 + math.sin(angle) * u.orbit_radius * 0.55 + u.drift_y

                        # Repel nearby UFOs to prevent clustering.
                        repel_x = repel_y = 0.0
                        for other in ufos:
                            if other is u:
                                continue
                            dx = u.x - other.x
                            dy = u.y - other.y
                            dist = max(1.0, math.hypot(dx, dy))
                            if dist < 90:
                                strength = (90 - dist) / 90
                                repel_x += dx / dist * strength * 130
                                repel_y += dy / dist * strength * 130
                        desired_x = max(30, min(w - 30, desired_x + repel_x))
                        desired_y = max(30, min(h - 35, desired_y + repel_y))
                        u.x += (desired_x - u.x) * min(1.0, 1.0 * dt)
                        u.y += (desired_y - u.y) * min(1.0, 1.0 * dt)
                        u.cooldown -= dt
                        if u.cooldown <= 0:
                            dx = target[0] - u.x
                            dy = target[1] - u.y
                            length = max(1.0, math.hypot(dx, dy))
                            speed = 190.0
                            shots.append(Shot(u.x, u.y + 12,
                                              dx / length * speed, dy / length * speed))
                            u.cooldown = random.uniform(0.9, 2.1)
                    draw_ufo(frame, u, elapsed)

                for s in shots[:]:
                    old_x, old_y = s.x, s.y
                    s.x += s.vx * dt
                    s.y += s.vy * dt
                    s.ttl -= dt
                    cv2.line(frame, (int(old_x), int(old_y)),
                             (int(s.x), int(s.y)), (30, 255, 255), 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(s.x), int(s.y)), 5, (0, 80, 255), -1)
                    hit = (box is not None and
                           box[0] <= s.x <= box[2] and box[1] <= s.y <= box[3])
                    if hit:
                        blasts.append(Blast(s.x, s.y))
                        shots.remove(s)
                        if invulnerable <= 0:
                            hp = max(0, hp - 10)
                            score += 1
                            invulnerable = 0.35
                    elif s.ttl <= 0 or not (-20 <= s.x <= w + 20 and -20 <= s.y <= h + 20):
                        shots.remove(s)

                invulnerable = max(0.0, invulnerable - dt)
            else:
                for u in ufos:
                    draw_ufo(frame, u, elapsed)

            for b in blasts[:]:
                b.age += dt
                radius = int(8 + b.age * 70)
                if b.age > 0.35:
                    blasts.remove(b)
                else:
                    cv2.circle(frame, (int(b.x), int(b.y)), radius,
                               (0, 180, 255), 3, cv2.LINE_AA)
                    cv2.circle(frame, (int(b.x), int(b.y)), max(2, radius // 3),
                               (0, 255, 255), -1, cv2.LINE_AA)

            status = 'TARGET' if target is not None and hp > 0 else 'NO TARGET'
            draw_hud(frame, hp, score, status)
            if hp <= 0:
                cv2.putText(frame, 'GAME OVER - PRESS R',
                            (max(15, w // 2 - 210), h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow('UFO Attack', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                ufos, shots, blasts, hp, score, invulnerable = reset()
                target, box = None, None
                last_seen = -100.0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
