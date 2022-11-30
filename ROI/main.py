import os
import pixellib
from pixellib.instance import custom_segmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 1, class_names= ['background','arm'])
segment_image.load_model("mask_rcnn_model.044-0.463592.h5")
for i in os.listdir('data/t1'):
    if i.endswith('.png'):
        data = segment_image.segmentImage("data/t1/"+i, show_bboxes=True, output_image_name="test.jpg")
        data = data[0]['masks']
        t=(len(data[0,0,:]))
        print(len(data[0,0,:]))

        data = data.reshape((1080, 1440,t))
        image = Image.open('data/t1/'+i)
        image = np.array(image)
        image = image.astype(int)
        image = np.multiply(data[:,:,0],image)
        np.putmask(image,image<2,255)
        plt.subplots(figsize=(14.4,10.8))
        plt.axis('off')
        plt.imshow(image,cmap='gray')
        #plt.show()
        plt.savefig('deleted_back/right/'+i)