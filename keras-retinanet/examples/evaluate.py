import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'inference', 'inference.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'defect'}
input_images = [] # Store resized versions of the images here.
img_path='/home/eric/data/aihub/VOCdevkit/VOC2007/JPEGImages'
images=os.listdir(img_path)
for single_image in images:
    single_image_path=os.path.join(img_path,single_image)
    image = read_image_bgr(single_image_path)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    txt_name=os.path.join('../predicted',single_image[:-4])+'.txt'
    with open(txt_name,'w') as f:
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            print(box,score)
            if score < 0.1:
                break
            f.write(labels_to_names[label])
            f.write(' '+str(score))
            f.write(' '+str(box[0]))
            f.write(' '+str(box[1]))
            f.write(' '+str(box[2]))
            f.write(' '+str(box[3]))
            f.write('\n')

# print("processing time: ", time.time() - start)
#     image_opencv=cv2.imread(single_image_path)

# image = read_image_bgr('/home/eric/data/aihub/VOCdevkit/VOC2007/JPEGImages/00008.BMP_block_35.jpg')

# copy to draw on
# draw = image.copy()
# draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# # preprocess image for network
# image = preprocess_image(image)
# image, scale = resize_image(image)

# # process image
# start = time.time()
# boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
# print("processing time: ", time.time() - start)

# # correct for image scale
# boxes /= scale

# # visualize detections
# for box, score, label in zip(boxes[0], scores[0], labels[0]):
#     # scores are sorted so we can break
#     print(box,score)
#     if score < 0.1:
#         break
        
#     color = label_color(label)
    
#     b = box.astype(int)
#     draw_box(draw, b, color=color)
    
#     caption = "{} {:.2f}".format(labels_to_names[label], score)
#     draw_caption(draw, b, caption)
    
# plt.figure(figsize=(12, 12))
# plt.axis('off')
# plt.imshow(draw)
# plt.show()