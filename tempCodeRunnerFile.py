
"""##Importing the necessary Libraries"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

"""##Reading the Image and conversion to Gray Scale"""

img = cv2.imread('D:\testing\\1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

"""##Noise Reduction and Edge Detection"""

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

"""##Tracking coordinates of Number Plate"""

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None

for contour in contours:
  approx = cv2.approxPolyDP(contour, 10, True)
  if len(approx) == 4:
    location = approx
    break

location

"""##Masking the Number Plates"""

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255,-1)
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

"""##Cropping the Image"""

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

"""##Extraction of Data from the Cropped Image"""

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result

"""##Displaying the Image along with Number"""

text = result[0][-2]
if text == "IND":
  text=result[1][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (approx[0][0][0], approx[2][0][1] + 60)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.text(200, -50, text, color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()