"""An implementation of the article - 'Fully Convolutional Network for Semantic Segmentation'.
The article can be found: https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_011.pdf"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Lambda

from conv2d_transpose import Conv2DTranspose
from keras import backend as K
from keras.models import Model, Sequential
from keras.utils import to_categorical
import tensorflow as tf


import numpy as np
from cv2 import imread
from os import listdir


TRAINING_LABEL_FILE = "ADEChallengeData2016/annotations/training"
TRAINING_DATA_FILE = "ADEChallengeData2016/images/training"


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2Dtranspose(prev, ks, filters, stride, padding='VALID'):
    shape = tf.shape(prev)
    W = weight_variable([ks, ks, shape[3], filters])
    upsampling = tf.nn.conv2d_transpose(prev, filter=W, strides=[1, stride, stride, 1], padding=padding)
    return upsampling


class FCN:

    def __init__(self, input_dim, num_labels, batch_size, learning_rate):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.model = None


    def load_model_from(self):
        pass


    def build_tf_model(self):
        input_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, None, 3])
        label_tensor = tf.placeholder(dtype = tf.float32, shape = [None, None, None, 151])
        self.x = input_tensor
        self.y = label_tensor
        shape = tf.shape(input_tensor)

        conv1_1 = tf.layers.conv2d(input_tensor ,kernel_size=3, filters=64, activation=tf.nn.relu)
        conv1_2 = tf.layers.conv2d(conv1_1,filters=64, kernel_size=3, activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv1_2, strides=2, pool_size=2)

        conv2_1 = tf.layers.conv2d(pool_1,filters=128, kernel_size=3, activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(conv2_1,filters=128, kernel_size=3, activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv2_2, strides=2, pool_size=2)

        conv3_1 = tf.layers.conv2d(pool_2,filters=256, kernel_size=3, activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(conv3_1,filters=256, kernel_size=3, activation=tf.nn.relu)
        conv3_3 = tf.layers.conv2d(conv3_2,filters=256, kernel_size=3, activation=tf.nn.relu)
        pool_3 = tf.layers.max_pooling2d(conv3_3, strides=2, pool_size=2)

        conv4_1 = tf.layers.conv2d(pool_3, filters=512, kernel_size=3, activation=tf.nn.relu)
        conv4_2 = tf.layers.conv2d(conv4_1, filters=512, kernel_size=3, activation=tf.nn.relu)
        conv4_3 = tf.layers.conv2d(conv4_2, filters=512, kernel_size=3, activation=tf.nn.relu)
        pool_4 = tf.layers.max_pooling2d(conv4_3, strides=2, pool_size=2)

        conv5_1 = tf.layers.conv2d(pool_4, filters=512, kernel_size=3, activation=tf.nn.relu)
        conv5_2 = tf.layers.conv2d(conv5_1, filters=512, kernel_size=3, activation=tf.nn.relu)
        conv5_3 = tf.layers.conv2d(conv5_2, filters=512, kernel_size=3, activation=tf.nn.relu)
        pool_5 = tf.layers.max_pooling2d(conv5_3, strides=2, pool_size=2)
        #till here is normal vgg
        #fully conv
        fc6 = tf.layers.conv2d(pool_5, filters=1024, kernel_size=7, activation=tf.nn.relu, padding='SAME')
        drop6 = tf.layers.dropout(fc6, 0.5)
        fc7 = tf.layers.conv2d(drop6, filters=1024, kernel_size=1, activation=tf.nn.relu)
        drop7 = tf.layers.dropout(fc7, 0.5)
        score_fr = tf.layers.conv2d(drop7, filters=self.num_labels, kernel_size=1, use_bias=False, name="score")
        conv0_up = tf.layers.conv2d_transpose(drop7, self.num_labels, 32, 16)
        # conv1_up = tf.layers.conv2d_transpose(conv0_up, 512, 4, 2)
        # conv2_up = tf.layers.conv2d_transpose(conv1_up, 256, 4, 2)
        # conv3_up = tf.layers.conv2d_transpose(conv2_up, 128, 4, 2)
        # conv4_up = tf.layers.conv2d_transpose(conv0_up, 151, 32, 16)
        upsampling = tf.image.resize_bilinear(conv0_up, shape[1:3])
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label_tensor, logits=upsampling)
        output = tf.argmax(tf.nn.softmax(upsampling), axis=3)
        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        return output, train_step, [conv0_up, pool_1, pool_2, pool_3, pool_4, pool_5], score_fr


    def build_network(self):
        """
        Creates a fully convolutional network with including upsampling.
        :return: the network
        """
        image_input = Input(shape=(None, None, 3))
        shape = K.shape(image_input)



        conv1_1 = Conv2D(kernel_size=3, filters=64, activation='relu')(image_input)
        conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv1_1)
        pool_1 = MaxPool2D(strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu')(pool_1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu')(conv2_1)
        pool_2 = MaxPool2D(strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu')(pool_2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu')(conv3_2)
        pool_3 = MaxPool2D(strides=2)(conv3_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu')(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)

        conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu')(pool_4)
        conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv5_2)
        pool_5 = MaxPool2D(strides=2)(conv5_3)

        #fully conv
        fc6 = Conv2D(filters=4096, kernel_size=7, activation='relu')(pool_5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=1, activation='relu')(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = Conv2D(filters=self.num_labels, kernel_size=1, use_bias=False)(drop7)
        upsampling = UpSampling2DBilinear(shape[1:3])(score_fr)
        # deconv = Conv2DTranspose(filters=self.num_labels, kernel_size=64, strides=32, activation='softmax')(score_fr)

        self.model = Model(inputs=image_input, outputs=upsampling)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        return upsampling

    def build_sequential_model(self):
        self.model = Sequential()

        image_input = Input(shape=(None, None, 3))
        shape = K.shape(image_input)

        conv1_1 = Conv2D(kernel_size=3, filters=64, activation='relu')(image_input)
        conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu')(conv1_1)
        pool_1 = MaxPool2D(strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu')(pool_1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu')(conv2_1)
        pool_2 = MaxPool2D(strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu')(pool_2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu')(conv3_2)
        pool_3 = MaxPool2D(strides=2)(conv3_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu')(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)

        conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu')(pool_4)
        conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu')(conv5_2)
        pool_5 = MaxPool2D(strides=2)(conv5_3)

        # fully conv
        fc6 = Conv2D(filters=4096, kernel_size=7, activation='relu')(pool_5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=1, activation='relu')(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = Conv2D(filters=self.num_labels, kernel_size=1, use_bias=False)(drop7)
        upsampling = UpSampling2DBilinear(shape[1:3])(score_fr)
        # deconv = Conv2DTranspose(filters=self.num_labels, kernel_size=64, strides=32, activation='softmax')(score_fr)

        self.model = Model(inputs=image_input, outputs=upsampling)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        return upsampling


    def train_keras(self, data, label):
        self.model.fit(data, label)


    def train_network(self, output, train_step, data, labels, layers):
        """Trains the network readjusting the weights accordingly"""
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(len(data)):
                out = sess.run([output, train_step] + layers, feed_dict={self.x: data[i], self.y: labels[i]})
                print(out)

        print("model successfully trained")

    def save_network(self, network, filename):
        pass

    def test_network(self, data, labels):
        if self.model:
            y = self.model.predict(data)

            print(y)



def load_data(sample_file, label_file, num_classes):
    data = load_images(sample_file)
    labels = load_images(label_file)
    for i, label in enumerate(labels):
        height, width, channels = labels[i].shape
        data[i] = data[i].reshape(1,height, width, 3)
        data[i] = data[i].astype(np.float32)
        labels[i] = np.reshape(to_categorical(label[:,:,0], num_classes), (1,height,width,num_classes))
    return data, labels

def load_images(folder, num_of_images=20):
    dataset = []
    for _file in listdir(folder)[:num_of_images]:
        img = imread(folder + "/" + _file)
        dataset.append(img)
    return dataset

def main():
    data, labels = load_data(TRAINING_DATA_FILE, TRAINING_LABEL_FILE, 151)
    net = FCN(1,151,1,1)
    net.build_network()
    for i in range(len(data)):
        net.train_keras(data[i].astype(np.float32), labels[i])

    #
    # output, train_step, up_layers, score = net.build_tf_model()
    # net.train_network(output, train_step, data,labels, layers=up_layers+[score])

def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))



def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                        1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights



def upsample_layer(bottom,
                   n_channels, name, upscale_factor):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)

        h = ((in_shape[1] - 1) * stride) + 1
        w = ((in_shape[2] - 1) * stride) + 1
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)

        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]

        weights = get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

    return deconv

if __name__ == '__main__':
    main()