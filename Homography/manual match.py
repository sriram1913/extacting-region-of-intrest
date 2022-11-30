import cv2
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt

img1 = im.open("leftact.jpg")  # Image to be aligned.
img2 = im.open("rightact.jpg")  # Reference image.

img1=img1.convert("RGBA")
height , width=img2.size

layer=np.zeros((width,height,3))

point=[495,471]
for i in range(0,10):
    for j in range(0, 5):
        layer[point[0]+i][point[1]+j]=(255,0,0)


imlayer=im.fromarray(layer,"RGBA")
new_im=im.blend(img1,imlayer,0.5)
new_im.save("overlay.png","PNG")