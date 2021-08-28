"""
Gradcam visualization ref modified from implementation by 
Author: [fchollet](https://twitter.com/fchollet)
"""

import cv2
import numpy as np
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model_modified import efficientdet_mod
from model import efficientdet
from utils import preprocess_image

def parse_args(args):
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser(description='Gradcam visualization script for Efficientdet.')

    parser.add_argument('--model-path', help='Path to trained model.', default='efficientdet-d1.h5')
    parser.add_argument('--phi', help='Hyper parameter phi', default=1, type=int, choices=(0, 1, 2, 3, 4, 5, 6))    
    parser.add_argument('--viz-cls', help='coco class to visualize', type=int, default=0)
    parser.add_argument('--img-path', help='image to visualize', default='sample\\person.jpg')

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)

    
def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    image_path = args.img_path#'sample\\person.jpg'
    top_pred_index = args.viz_cls#choose which class detection to watch 0=person
    model_path = args.model_path#'efficientdet-d1.h5'
    phi = args.phi#1

    weighted_bifpn = True
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    num_classes = 90
    score_threshold = 0.3

    #load modified efficientdet 
    #get the last conv layer before inference and prediction models
    conv_layer_out,pred_models = efficientdet_mod(phi=phi,
                        weighted_bifpn=weighted_bifpn,
                        num_classes=num_classes,
                        score_threshold=score_threshold)
    num_layers = len(conv_layer_out)
    for i in range(num_layers):
        conv_layer_out[i].load_weights(model_path, by_name=True)  
        pred_models[i].load_weights(model_path, by_name=True)
        
    image = cv2.imread(image_path)
    src_image = image.copy()
    # BGR -> RGB
    image = image[:, :, ::-1]
    h, w = image.shape[:2]
    image, scale = preprocess_image(image, image_size=image_size)

    #to be used for display
    img = keras.preprocessing.image.img_to_array(image)
    image = [np.expand_dims(image, axis=0)]

    #Create an combined image with all gradcams from different layers
    out_image = np.zeros((height*3,width*2,3), np.uint8)
    
    out_image[0:height,0:width,:] = src_image
    
    for i in range(0,num_layers):
        with tf.GradientTape() as tape:
            last_conv_layer_output = conv_layer_out[i](image)
            preds = pred_models[i](last_conv_layer_output)
            top_class_channel = preds[:, :, top_pred_index]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # compute the guided gradients
        castConvOutputs = tf.math.abs(last_conv_layer_output)
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = last_conv_layer_output[0]
        guidedGrads = guidedGrads[0]
        
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        
        heatmap = cv2.resize(cam.numpy(), (image_sizes[phi], image_sizes[phi]))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        eps = 0.000001
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        
        # We use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((image_sizes[phi], image_sizes[phi]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * 0.3 + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Display Grad CAM
        #plt.matshow(superimposed_img)
        #plt.show()
        
if __name__ == '__main__':
    main()
