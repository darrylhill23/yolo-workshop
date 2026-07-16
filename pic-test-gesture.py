#@markdown We implemented some functions to visualize the gesture recognition results. <br/> Run the following cell to activate the functions.
from matplotlib import pyplot as plt
import mediapipe as mp
import urllib.request

import cv2

import math


plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]

    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows
    rows = max(1, rows)
    cols = max(1, cols)

    TARGET_W, TARGET_H = 320, 320

    grid_images = []

    for i in range(rows * cols):
        if i >= len(images):
            break

        img = images[i].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        g = gestures[i]
        label = f"{g.category_name} ({g.score:.2f})"

        # draw landmarks (CORRECT MEDIA PIPE STRUCTURE)
        for hand_landmarks in multi_hand_landmarks_list[i]:

            for connection in mp_hands.HAND_CONNECTIONS:

                start = hand_landmarks[connection.start]
                end = hand_landmarks[connection.end]


                h, w, _ = img.shape

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # optional: draw points
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        img = cv2.resize(img, (TARGET_W, TARGET_H))
        grid_images.append(img)

    # pad grid if needed
    while len(grid_images) < rows * cols:
        grid_images.append(
            (255 * np.ones((TARGET_H, TARGET_W, 3), dtype=np.uint8))
        )

    rows_imgs = []
    for r in range(rows):
        row = grid_images[r * cols:(r + 1) * cols]
        rows_imgs.append(cv2.hconcat(row))

    final_grid = cv2.vconcat(rows_imgs)

    cv2.imshow("Gestures", final_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
  urllib.request.urlretrieve(url, name)

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480



# Preview the images.
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='gesture-models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

images = []
results = []
for image_file_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_file_name)

  # STEP 4: Recognize gestures in the input image.
  recognition_result = recognizer.recognize(image)

  # STEP 5: Process the result. In this case, visualize it.
  images.append(image)
  top_gesture = recognition_result.gestures[0][0]
  hand_landmarks = recognition_result.hand_landmarks
  results.append((top_gesture, hand_landmarks))

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
