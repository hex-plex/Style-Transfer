import numpy as np
import tensorflow as tf
from tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
import cv2

from scipy.optimize import fmin_l_bfgs_b

DEBUG=False

ITERATIONS = 15
CHANNELS = 3
IMAGE_SIZE = 500
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
STYLE_WIDTH = 500
STYLE_HEIGHT = 400
IMAGENET_MEAN_RGB_VALUES = [123.68, 116.779, 103.939]
CONTENT_WEIGHT = 0.02
STYLE_WEIGHT = 4.5
TOTAL_VARIATION_WEIGHT = 0.995
TOTAL_VARIATION_LOSS_FACTOR = 1.25

input_image_array = cv2.imread("content.jpg").astype(np.float64)
input_image_array = cv2.resize(input_image_array,(400,300))
input_image_array = np.expand_dims(input_image_array,axis=0)
input_image_array[:,:,:,0] -= IMAGENET_MEAN_RGB_VALUES[0]
input_image_array[:,:,:,1] -= IMAGENET_MEAN_RGB_VALUES[1]
input_image_array[:,:,:,2] -= IMAGENET_MEAN_RGB_VALUES[2]

style_image_array = cv2.imread("style.jpg").astype(np.float64)
style_image_array = cv2.resize(style_image_array,(400,300))
style_image_array = np.expand_dims(style_image_array,axis=0)
style_image_array[:,:,:,0] -= IMAGENET_MEAN_RGB_VALUES[0]
style_image_array[:,:,:,1] -= IMAGENET_MEAN_RGB_VALUES[1]
style_image_array[:,:,:,2] -= IMAGENET_MEAN_RGB_VALUES[2]

input_image = K.variable(input_image_array)
style_image = K.variable(style_image_array)
combination_image = K.placeholder(shape=(1,IMAGE_HEIGHT, IMAGE_WIDTH, 3),name="c_img")

input_tensor =  K.concatenate([input_image, style_image, combination_image], axis=0)

model = VGG16(input_tensor=input_tensor, include_top=False)


def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = "block2_conv2"
layer_features = layers[content_layer]
content_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]

loss = K.variable(0.)
loss = loss + CONTENT_WEIGHT*content_loss(content_image_features, combination_features)

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def compute_style_loss(style, combination):
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = IMAGE_HEIGHT*IMAGE_WIDTH
    return K.sum(K.square(style-combination)) / (4. * (CHANNELS**2) * (size**2))

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
for layer_name in style_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    style_loss = compute_style_loss(style_features,combination_features)
    loss = loss + (STYLE_WEIGHT / len(style_layers)) * style_loss

def total_variation_loss(x):
    a = K.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, 1:, :IMAGE_WIDTH-1, :])
    b = K.square(x[:, :IMAGE_HEIGHT-1, :IMAGE_WIDTH-1, :] - x[:, :IMAGE_HEIGHT-1, 1:, :])

    return K.sum(K.pow(a+b, TOTAL_VARIATION_LOSS_FACTOR))

loss = loss + TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image)


def evaluate_loss_and_gradients(x):
    global loss, combination_image
    x = x.reshape((1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    outputs= [loss, K.gradients(loss ,[combination_image])[0]]
    if DEBUG:
        print("loss",loss)
        print("combination_image",combination_image)
        print("x",x)
        print("grad",outputs[1],"end")
    outs = K.function([combination_image],outputs)(K.variable(x,name="inti"))
    los = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return los , gradients

class Evaluator:

    def loss(self,x):
        los, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return los

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)) - 128.

for i in range(ITERATIONS):
    x, los, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients , maxfun=20)
    y = x.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    y[:,:,0] +=IMAGENET_MEAN_RGB_VALUES[0]
    y[:,:,1] +=IMAGENET_MEAN_RGB_VALUES[1]
    y[:,:,2] +=IMAGENET_MEAN_RGB_VALUES[2]
    y = np.clip(y, 0, 255).astype("uint8")
    cv2.imwrite("outputs/combined"+str(i)+".jpg", y)
    print("The "+str(i)+" Iteration has completed with loss:"+str(los))
