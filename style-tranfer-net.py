import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import cv2

import scipy.optimize import fmin_l_bfgs_b

ITERATIONS = 15
CHANNELS = 3
IMAGE_SIZE = 500
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 3000
STYLE_WIDTH = 500
STYLE_HEIGHT = 400
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

model = VGG16(input_tensor=input_tensor, include_top=False)


def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = "block2_conv2"
layer_features = layers[content_layer]
content_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]

loss = backend.variable(0.)
loss += CONTENT_WEIGHT*content_loss(content_image_features, combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2,0,1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT*IMAGE_WIDTH
    return backend.sum(backend.square(style-combination)) / (4. * (CHANNELS**2) * (size**2))

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    style_loss = compute_style_loss(style_features,combination_features)
    loss += (STYLE_WEIGHT / len(style_layers)) * style_loss

def total_variation_loss(x):
    a = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = backend.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGTH-1, 1:, :])

    return backend.sum(backend.pow(a+b, TOTAL_VARIATION_LOSS))

loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)

outputs = [loss]
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outs = backend.function([combination_image],outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:

    def loss(self,x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1,IMAGE_HEIGHT,IMAGE_WIDTH, 3)) -128.

for i in range(ITERATIONS):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients , maxfun=20)
    y = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    y[:,:,0] +=IMAGENET_MEAN_RGB_VALUES[0]
    y[:,:,1] +=IMAGENET_MEAN_RGB_VALUES[1]
    y[:,:,2] +=IMAGENET_MEAN_RGB_VALUES[2]
    y = np.clip(y, 0, 255).astype("uint8")
    cv2.imwrite("combined"+str(i)+".jpg", y)
    
