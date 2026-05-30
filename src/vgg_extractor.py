import numpy as np
import tensorflow as tf
from settings import VGG_MEAN


class VGGExtractor(tf.keras.Model):
    def __init__(self):
        super(VGGExtractor, self).__init__()

        self.data_dict = np.load("model/vgg16partial.npy", allow_pickle=True).item()
        self.build_model()
        self.data_dict = None

    def build_model(self):
        self.conv1_1 = self.conv_layer("conv1_1")
        self.conv1_2 = self.conv_layer("conv1_2")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv2_1 = self.conv_layer("conv2_1")
        self.conv2_2 = self.conv_layer("conv2_2")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv3_1 = self.conv_layer("conv3_1")
        self.conv3_2 = self.conv_layer("conv3_2")
        self.conv3_3 = self.conv_layer("conv3_3")
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv4_1 = self.conv_layer("conv4_1")
        self.conv4_2 = self.conv_layer("conv4_2")
        self.conv4_3 = self.conv_layer("conv4_3")
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        self.conv5_1 = self.conv_layer("conv5_1")
        self.pool5_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    def conv_layer(self, name):
        kernel = self.data_dict[name][0]
        bias = self.data_dict[name][1]

        layer = tf.keras.layers.Conv2D(
            filters=kernel.shape[3],
            kernel_size=(kernel.shape[0], kernel.shape[1]),
            padding='same',
            activation='relu',
            name=name,
            kernel_initializer=tf.keras.initializers.Constant(kernel),
            bias_initializer=tf.keras.initializers.Constant(bias),
            trainable=False
        )
        return layer

    def call(self, bgr):
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr)
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        x1_1 = self.conv1_1(bgr)
        x1_2 = self.conv1_2(x1_1)
        pool1 = self.pool1(x1_2)

        x2_1 = self.conv2_1(pool1)
        x2_2 = self.conv2_2(x2_1)
        pool2 = self.pool2(x2_2)

        x3_1 = self.conv3_1(pool2)
        x3_2 = self.conv3_2(x3_1)
        x3_3 = self.conv3_3(x3_2)
        pool3 = self.pool3(x3_3)

        x4_1 = self.conv4_1(pool3)
        x4_2 = self.conv4_2(x4_1)
        x4_3 = self.conv4_3(x4_2)
        pool4 = self.pool4(x4_3)

        x5_1 = self.conv5_1(pool4)
        pool5_1 = self.pool5_1(x5_1)

        return pool3, pool4, pool5_1
