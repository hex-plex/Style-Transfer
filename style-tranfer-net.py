import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
ITERATIONS = 15
CHANNELS = 3
IMAGE_SIZE = 500
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 3000
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

input_image_array = cv2.imread("content.jpg")
input_image_array = np.expand_dims(input_image_array,axis=0)
input_image_array[:,:,:,0] -= IMAGENET_MEAN_RGB_VALUES[0]
input_image_array[:,:,:,1] -= IMAGENET_MEAN_RGB_VALUES[1]
input_image_array[:,:,:,2] -= IMAGENET_MEAN_RGB_VALUES[2]

style_image_array = cv2.imread("style.jpg")
style_image_array = np.expand_dims(style_image_array,axis=0)
style_image_array[:,:,:,0] -= IMAGENET_MEAN_RGB_VALUES[0]
style_image_array[:,:,:,1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array[:,:,:,2] -= IMAGENET_MEAN_RGB_VALUES[2]

input_image = backend.variable(input_image_array)
style_image = backend.variable(style_image_array)
combination_image = backend.variable((1,IMAGE_HEIGHT, IMAGE_WIDTH, 3))

input_tensor =  backend.concatenate([input_image, style_image, combination_image])
