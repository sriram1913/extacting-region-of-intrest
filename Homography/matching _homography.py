import cv2
import numpy as np

img1 = cv2.imread("left0.png", cv2.IMREAD_GRAYSCALE)  # Image to be aligned.
img2 = cv2.imread("right0.png", cv2.IMREAD_GRAYSCALE)  # Reference image.

height, width = img2.shape

p1 = [[668, 284], [528, 282], [512, 321], [628, 430], [862, 440], [881, 399], [779, 341]];
p2 = [[587, 295], [485, 298], [467, 334], [564, 425], [773, 427], [778, 395], [685, 344]];
p1 = np.array(p1);
p2 = np.array(p2);
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

transformed_img = cv2.warpPerspective(img1,
                                      homography, (width, height))

cv2.imwrite('output_manual-extr.jpg', transformed_img)
