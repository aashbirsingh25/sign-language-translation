#!/usr/bin/env python3
import argparse
import time
from collections import deque, Counter

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

from model import create_model
from processing import square_pad, preprocess_for_vgg

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='model name (vgg16,resnet,inception,xception,mobilenet)')
ap.add_argument('-w', '--weights', default=None, help='path to classification weights (optional)')
args = ap.parse_args()

if mp is None:
    raise ImportError('mediapipe is required. Install with: pip install mediapipe')

clf = create_model(model=args.model, model_weights_path=args.weights)

import string
label_dict = {i: c for i, c in enumerate(string.ascii_uppercase)}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

WINDOW_SIZE = 7
pred_window = deque(maxlen=WINDOW_SIZE)

NO_HAND_MAX_FRAMES = 20
no_hand_counter = 0
assembled = []
current_word = ''

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    start = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frames += 1
        h, w = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        hand_detected = False
        if results.multi_hand_landmarks:
            hand_detected = True
            no_hand_counter = 0
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(max(0, min(xs) * w) - 20)
            x_max = int(min(w, max(xs) * w) + 20)
            y_min = int(max(0, min(ys) * h) - 20)
            y_max = int(min(h, max(ys) * h) + 20)

            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                crop = frame.copy()

            crop = square_pad(crop)
            x_in = preprocess_for_vgg(crop)

            preds = clf.predict(x_in, batch_size=1, verbose=0)
            conf = float(np.max(preds))
            top = int(np.argmax(preds))

            if conf >= 0.45:
                pred_letter = label_dict[top]
                pred_window.append(pred_letter)
            else:
                pred_window.append(None)

            most_common = Counter(pred_window).most_common(1)
            if most_common:
                letter, _ = most_common[0]
                if letter is not None:
                    if len(current_word) == 0 or current_word[-1] != letter:
                        current_word += letter
        else:
            no_hand_counter += 1
            pred_window.append(None)

        if no_hand_counter > NO_HAND_MAX_FRAMES and len(current_word) > 0:
            assembled.append(current_word)
            current_word = ''

        cv2.imshow('Mediapipe Translator', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            assembled = []
            current_word = ''

    end = time.time()
    fps = frames / (end - start)
    print('[INFO] approx FPS: {:.2f}'.format(fps))

cap.release()
cv2.destroyAllWindows()
