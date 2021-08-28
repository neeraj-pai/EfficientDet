"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2020/05/14
Description: How to obtain a class activation heatmap for an image classification model.
"""
"""
Adapted from Deep Learning with Python (2017).
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#efficientdet part
import cv2
import json
import numpy as np
import os
import time
import glob

from model_modified import efficientdet_mod
from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

#load modified efficientdet 
phi = 1
weighted_bifpn = True
model_path = 'efficientdet-d1.h5'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
num_classes = 90
score_threshold = 0.3

class_model,eff_model = efficientdet(phi=phi,
                    weighted_bifpn=weighted_bifpn,
                    num_classes=num_classes,
                    score_threshold=score_threshold)
conv_layer_out,pred_models = efficientdet_mod(phi=phi,
                    weighted_bifpn=weighted_bifpn,
                    num_classes=num_classes,
                    score_threshold=score_threshold)
eff_model.load_weights(model_path, by_name=True)  
class_model.load_weights(model_path, by_name=True)                    
for i in range(5):
    conv_layer_out[i].load_weights(model_path, by_name=True)  
    pred_models[i].load_weights(model_path, by_name=True)
    
#0-57600=6400=80x80
#1-14400=1600=40x40
#2-3600=400=20x20
#3-900=100=10x10
#4-225=25=5x5

image_path = 'sample\\test_image.jpg'
image = cv2.imread(image_path)
src_image = image.copy()
# BGR -> RGB
image = image[:, :, ::-1]
h, w = image.shape[:2]

image, scale = preprocess_image(image, image_size=image_size)

#to be used for display
img = keras.preprocessing.image.img_to_array(image)

image = [np.expand_dims(image, axis=0)]
'''
#print(conv_layer_output[0],conv_layer_output[2])
boxes, scores, labels = eff_model.predict_on_batch(image)
classification,regression = class_model.predict_on_batch(image)
boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
#test if we can get last conv layer outputs for all layers
conv_layer_output1 = conv_layer_out[0].predict_on_batch(image)
class1_output = pred_models[0].predict_on_batch(conv_layer_output1)
conv_layer_output2 = conv_layer_out[1].predict_on_batch(image)
class2_output = pred_models[1].predict_on_batch(conv_layer_output2)
conv_layer_output3 = conv_layer_out[2].predict_on_batch(image)
class3_output = pred_models[2].predict_on_batch(conv_layer_output3)

#print(classification.shape,conv_layer_output[0].shape,class1_output.shape)
print(classification[0,12,0:4],class1_output[0,12,0:4])
print(classification[0,80*80*9+12,0:4],class2_output[0,12,0:4])
print(classification[0,80*80*9+40*40*9+12,0:4],class3_output[0,12,0:4])
'''



for i in range(0,5):
    with tf.GradientTape() as tape:
        last_conv_layer_output = conv_layer_out[i](image)
        #print(last_conv_layer_output[0,0,0:4])
        preds = pred_models[i](last_conv_layer_output)
        #print(preds[0,12,0:4])
        top_pred_index = 0#choose which class detection to watch 0=person
        top_class_channel = preds[:, :, top_pred_index]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    #print(np.max(grads),np.min(grads),np.mean(grads))

    #print(last_conv_layer_output[0,0,0:4])
    # compute the guided gradients
    #castConvOutputs = tf.cast(last_conv_layer_output > 0, "float32")#becoming zero here
    castConvOutputs = tf.math.abs(last_conv_layer_output)
    #print(np.max(castConvOutputs),np.min(castConvOutputs),np.mean(castConvOutputs))
    castGrads = tf.cast(grads > 0, "float32")
    #print(np.max(castGrads),np.min(castGrads),np.mean(castGrads))
    guidedGrads = castConvOutputs * castGrads * grads
    #print(np.max(guidedGrads),np.min(guidedGrads),np.mean(guidedGrads))
    convOutputs = last_conv_layer_output[0]
    guidedGrads = guidedGrads[0]
    
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    #print(np.max(weights),np.min(weights),np.mean(weights))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    #print(np.max(cam),np.min(cam),np.mean(cam))
    
    #(w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (640, 640))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    eps = 0.000001
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    #print(np.max(heatmap),np.min(heatmap),np.mean(heatmap))
    #plt.matshow(heatmap)
    #plt.show()
    
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((640, 640))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.2 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    plt.matshow(superimposed_img)
    plt.show()
    
