from PIL import Image as im
import matplotlib.pyplot as plt

img1 = im.open("output_manual-extr.jpg")  # Image to be aligned.
img2 = im.open("right0.png")  # Reference image.

img1=img1.convert("RGBA")
img2=img2.convert("RGBA")

new_im=im.blend(img1,img2,0.9)
new_im.save("overlay_matched-2.png","PNG")