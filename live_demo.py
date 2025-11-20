#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIVE DEMO
This script loads a pre-trained model (for best results use pre-trained weights for classification block)
and classifies American Sign Language finger spelling frame-by-frame in real-time
"""

import string
import cv2
import time
from processing import square_pad, preprocess_for_vgg
from model import create_model
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default=None,
                help="path to the model weights")
required_ap = ap.add_argument_group('required arguments')
required_ap.add_argument("-m", "--model",
                         type=str, default="resnet", required=True,
                         help="name of pre-trained network to use")
args = vars(ap.parse_args())


# ====== Create model for real-time classification ======
# =======================================================

# Map model names to classes
MODELS = ["resnet", "vgg16", "inception", "xception", "mobilenet"]

if args["model"] not in MODELS:
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

# Create pre-trained model + classification block, with or without pre-trained weights
my_model = create_model(model=args["model"],
                        model_weights_path=args["weights"])

# Dictionary to convert numerical classes to alphabet
label_dict = {pos: letter
              for pos, letter in enumerate(string.ascii_uppercase)}


# ====================== Live loop ======================
# =======================================================

video_capture = cv2.VideoCapture(0)

fps = 0
start = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    fps += 1

    # Guard: ensure a valid frame
    if frame is None:
        # skip this iteration if no frame captured
        continue

    # Compute frame size (width, height) dynamically
    height, width = frame.shape[:2]

    # Define a region of interest (ROI) relative to the frame size so the
    # rectangle and crop work for different camera resolutions.
    # We'll create a centered rectangle sized as a fraction of the frame.
    roi_w = int(width * 0.35)
    roi_h = int(height * 0.65)

    x = max(0, width // 2 - roi_w // 2)
    y = max(0, height // 2 - roi_h // 2)

    # Ensure ROI stays within the frame bounds
    x2 = min(width, x + roi_w)
    y2 = min(height, y + roi_h)

    cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 0), 3)

    # Crop + process captured frame using the dynamic ROI
    hand = frame[y:y2, x:x2]
    hand = square_pad(hand)
    hand = preprocess_for_vgg(hand)

    # Make prediction
    my_predict = my_model.predict(hand,
                                  batch_size=1,
                                  verbose=0)

    # Predict letter
    top_prd = np.argmax(my_predict)

    # Only display predictions with probabilities greater than 0.5
    if np.max(my_predict) >= 0.50:

        prediction_result = label_dict[top_prd]
        preds_list = np.argsort(my_predict)[0]
        pred_2 = label_dict[preds_list[-2]]
        pred_3 = label_dict[preds_list[-3]]

    # Use current frame size for annotation placement
    # (height and width already computed above)
    # Scale font sizes and thickness with frame width so text is readable
    scale_factor = max(0.5, width / 1280.0)
    main_font_scale = max(1.0, 6.0 * scale_factor)
    small_font_scale = max(0.7, 2.0 * scale_factor)
    main_thickness = max(2, int(6 * scale_factor))
    small_thickness = max(1, int(2 * scale_factor))

    if np.max(my_predict) >= 0.50:
        # Position main prediction to the right of the ROI, slightly above center
        main_x = min(width - 10, x2 + 10)
        main_y = max(30, y + 40)

        cv2.putText(frame, text=prediction_result,
                    org=(main_x, main_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=main_font_scale, color=(255, 255, 0),
                    thickness=main_thickness, lineType=cv2.LINE_AA)

        # Second and third predictions displayed below the ROI, left and right
        sec_x = x + 10
        sec_y = min(height - 10, y2 + 40)
        thr_x = max(10, x2 - 120)
        thr_y = sec_y

        cv2.putText(frame, text=pred_2,
                    org=(sec_x, sec_y),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=small_font_scale, color=(0, 0, 255),
                    thickness=small_thickness, lineType=cv2.LINE_AA)

        cv2.putText(frame, text=pred_3,
                    org=(thr_x, thr_y),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=small_font_scale, color=(0, 0, 255),
                    thickness=small_thickness, lineType=cv2.LINE_AA)
    else:
        # Optional: show a small notice that no confident prediction was made
        notice = "..."
        cv2.putText(frame, text=notice,
                    org=(x + 10, y + 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8 * scale_factor, color=(200, 200, 200),
                    thickness=1, lineType=cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit live loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Calculate frames per second
end = time.time()
FPS = fps/(end-start)
print("[INFO] approx. FPS: {:.2f}".format(FPS))

# Release the capture
video_capture.release()
cv2.destroyAllWindows()

