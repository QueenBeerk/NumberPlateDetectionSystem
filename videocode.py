import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
from easyocr import Reader


reader = Reader(['en'])

cap = cv2.VideoCapture('/content/1.mp4')

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply Canny edge detection
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]

        # Use EasyOCR to read text
        result = reader.readtext(cropped_image)

        if result:
            text = result[0][-2]

            # Overlay text on original image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

    out.write(img)

# Release video capture and writer
cap.release()
out.release()