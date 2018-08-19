"""An implementation of the article - 'Fully Convolutional Network for Semantic Segmentation'.
The article can be found: https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_011.pdf"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# to suppress tensorflow messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1
# set_session(tf.Session(config=config))

from pympler import asizeof
# from keras.preprocessing.image import ImageDataGenerator

from utils import load_data_generator, as_keras_metric, new_except_hook, get_model_memory_usage, get_bilinear_filter, computeIoU, plot_images, crop2d
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, ZeroPadding2D, Conv2DTranspose, Activation, Add
from os.path import join
import time
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import gc
import numpy as np

import atexit
import sys


TRAINING_LABEL_FILE = "ADEChallengeData2016/annotations/training"
TRAINING_DATA_FILE = "ADEChallengeData2016/images/training"
VALIDATION_LABEL_FILE = "ADEChallengeData2016/annotations/validation"
VALIDATION_DATA_FILE = "ADEChallengeData2016/images/validation"
PRETRAINED_WEIGHTS_FILE = "vgg16.npy"
WEIGHT_FOLDER = 'weights'
WEIGHT_FILE = "weights.hdf5"
LOAD_FILE = None
NUM_OF_CLASSES = 151
SAMPLES_PER_EPOCH = 20210


@as_keras_metric
def mean_iou( y_true, y_pred, num_classes=NUM_OF_CLASSES):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

def create_mean_iou(y_true, y_pred, num_classes=NUM_OF_CLASSES):
    def my_mean_iou(y_true, y_pred):
        return computeIoU(y_pred, y_true, num_classes)
    return my_mean_iou

class FCN:

    def __init__(self, num_labels, batch_size=1, learning_rate=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.model = None

        self.files_missed = []
        self.min_data = np.inf
        self.max_data = -np.inf
        self.min_res = np.inf
        self.max_res = -np.inf


    def load_model_from(self, weight_file):
        # load json and create model
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(join(WEIGHT_FOLDER, weight_file))
        print("Loaded model from disk")



    def build_network(self):
        """
        Creates a fully convolutional network with including upsampling.
        :return: the network
        """

        image_input = Input(shape=(None, None, 3))
        input_shape = K.shape(image_input)

        # padding = ZeroPadding2D(100, input_shape=(None, None, 3))(image_input)

        conv1_1 = Conv2D(kernel_size=3, filters=64, activation='relu', name="conv1_1", padding='same')(image_input)
        conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name="conv1_2", padding='same')(conv1_1)
        pool_1 = MaxPool2D(strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_1", padding='same')(pool_1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_2", padding='same')(conv2_1)
        pool_2 = MaxPool2D(strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_1", padding='same')(pool_2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_2", padding='same')(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_3", padding='same')(conv3_2)
        pool_3 = MaxPool2D(strides=2)(conv3_3)
        self.pool3 = Model(input=image_input, output=pool_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_1", padding='same')(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_2", padding='same')(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_3", padding='same')(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)
        self.pool4 = Model(input=image_input, output=pool_4)

        conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_1", padding='same')(pool_4)
        conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_2", padding='same')(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_3", padding='same')(conv5_2)
        pool_5 = MaxPool2D(strides=2)(conv5_3)

        #fully conv
        fc6 = Conv2D(filters=4096, kernel_size=7, activation='relu', name="fc6", padding='same')(pool_5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=1, activation='relu', name="fc7", padding='same')(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = Conv2D(filters=self.num_labels, kernel_size=1, use_bias=False, name="fc8", padding='same')(drop7)
        self.score_fr = Model(input=image_input, output=score_fr)
        deconv1 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv1")(score_fr) #deconv 32
        self.deconv1 = Model(input=image_input, output=deconv1)
        crop_deconv1 = crop2d(pool_4)(deconv1)
        self.crop1 = Model(input=image_input, output=crop_deconv1)
        skip1 = Conv2D(filters=self.num_labels, kernel_size=1, padding='same', activation=None, name='score_pool4')(pool_4)
        self.skip1 = Model(input=image_input, output=skip1)
        add_pool_4 = Add()([skip1, crop_deconv1])

        self.add_pool4 = Model(input=image_input, output=add_pool_4)
        deconv2 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv2")(add_pool_4)
        self.deconv2 = Model(input=image_input, output=deconv2)
        crop_deconv2 = crop2d(pool_3)(deconv2)
        self.crop2 = Model(input=image_input, output=crop_deconv2)

        skip2 = Conv2D(filters=self.num_labels, kernel_size=1, activation=None, name="score_pool3", padding="same")(pool_3)
        self.skip2 = Model(input=image_input, output=skip2)
        add_pool_3 = Add()([skip2, crop_deconv2])

        deconv3 = Conv2DTranspose(filters= self.num_labels, kernel_size=16, strides=8, activation=None, name="final")(add_pool_3)
        crop_deconv3 = crop2d(image_input)(deconv3)


        output = Activation('softmax', name="softmax_layer")(crop_deconv3)
        self.model = Model(inputs=image_input, outputs=output)
        self.set_keras_weights()
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy', mean_iou])
        return output


    def set_keras_weights(self):
        weights = np.load("vgg16.npy", encoding='latin1').item()
        weights['fc6'][0] = weights['fc6'][0].reshape(7,7,512,4096)
        weights['fc7'][0] = weights['fc7'][0].reshape(1, 1, 4096, 4096)
        for layer in self.model.layers:
            if layer.name in weights:
                if layer.name == 'fc8':
                    continue
                layer.set_weights(weights[layer.name])
            if layer.name in ['deconv1', "deconv2", "final"]:
                current_weights = layer.get_weights()
                bilinear_weights = get_bilinear_filter(current_weights[0].shape, layer.strides[0])
                layer.set_weights([bilinear_weights, current_weights[1]])



    def train_keras(self, data_file, label_file, test_data_file=VALIDATION_DATA_FILE, test_label_file=VALIDATION_LABEL_FILE):
        data_generator = load_data_generator(data_file, label_file, num_classes=self.num_labels, preload=9, batch_size=self.batch_size, shuffle=True, return_with_selection=False)
        test_generator = load_data_generator(test_data_file, test_label_file, num_classes=self.num_labels, preload=10, batch_size=self.batch_size, shuffle=True, return_with_selection=False)
        filepath = os.path.join(WEIGHT_FOLDER, WEIGHT_FILE)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callback_list = [checkpoint]
        self.model.fit_generator(data_generator, max_queue_size=4, steps_per_epoch=20, validation_data=test_generator, verbose=1, validation_steps=5, epochs=10000)


    def train_keras_gpu(self, data_file, label_file, test_interval=10, test_data_file=VALIDATION_DATA_FILE, test_label_file=VALIDATION_LABEL_FILE):
        file_num = 1
        mean_IoU = 0
        IoU = []
        second_past = time.time()
        data_generator = load_data_generator(data_file, label_file, num_classes=self.num_labels, preload=10, batch_size=self.batch_size, shuffle=True)
        test_generator = load_data_generator(test_data_file, test_label_file, num_classes=self.num_labels, preload=5, batch_size=self.batch_size, shuffle=True)
        for data, label, selection in data_generator:

            size_of_data = asizeof.asizeof(data)
            resolution_data = np.size(data)
            try:
                if selection in self.files_missed:
                    with tf.device('/cpu:0'):
                        print("using cpu")
                        self.model.fit(data, label, verbose=2)
                else:
                    with tf.device('/gpu:0'):
                        print("using gpu")
                        self.model.fit(data, label, verbose=2)

                    if self.max_res < resolution_data:
                        self.max_res = resolution_data
                        print("new max resolution : ", resolution_data)
                    if self.max_data < size_of_data:
                        print("new maxsize : ", size_of_data)
                        self.max_data = size_of_data
            except tf.errors.ResourceExhaustedError as re:
                print("error occured using cpu")
                print(re)
                if self.min_res > resolution_data:
                    self.min_res = resolution_data
                    print("new minresolution : ", resolution_data)
                if self.min_data > size_of_data:
                    print("new minsize : ", size_of_data)
                    self.min_data = size_of_data
                self.files_missed += list(selection)
            except KeyboardInterrupt as kb:
                break
            except Exception as e:
                print(e)


            del data
            del label
            gc.collect()
            print("Images Processed : ", file_num)

            if file_num % test_interval == 0:
                IoU += [self.test_network(test_generator, num_of_batches=5)]
                mean_IoU = (IoU[-1]*((file_num/test_interval) - 1) + mean_IoU) / (file_num/test_interval)
                real_IoU = np.mean(np.array(IoU))
                print("mean IOU: ", mean_IoU, real_IoU)

            if file_num % 100 == 0:
                self.save_network('weights_{0}'.format(real_IoU))
            print("Mean seconds  per image : {0}".format((time.time()-second_past)/file_num))
            file_num += 1

        print("Number of files missed", self.files_missed, "\n")

        #steps per epoch = num_samples/batch size
        # self.model.fit_generator(data_generator, steps_per_epoch=SAMPLES_PER_EPOCH, max_queue_size=1)


    def save_network(self, file_name='weights'):
        self.model.save_weights(join(WEIGHT_FOLDER,"{0}.h5".format(file_name)))
        print("weights Saved")
        # model_json = self.model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)


    def on_exit(self):
        a = input("save weights ? (y/n) : \n")
        if a == 'y':
            self.model.save_weights(join(WEIGHT_FOLDER, "auto_exit_weights.h5"))
            print("weights saved")
        print("Number of files missed during run : ", len(self.files_missed), "\n")
        print(self.files_missed)
        print("Max data size without exception: ", self.max_data)
        print("Max img resolution without exception: ", self.max_res)
        print("Min data size that caused exception: ", self.min_data)
        print("Min img resolution that caused exception: ", self.min_res)

    def test_network(self, data_generator, num_of_batches=5):
        IoU = []
        if self.model:
            i = 0
            while i < num_of_batches:
                data, label, selection = next(data_generator)
                i += 1
                try:
                    y_pred = self.model.predict(data)
                except Exception as error:
                    print(error)
                    continue
                IoU += [tf.metrics.mean_iou(label, y_pred, self.num_labels)]
                # IoU += [computeIoU(y_pred,label, self.num_labels)]
                print(IoU)
            mean = np.mean(np.array(IoU))
            print(mean)
            return mean



def main():
    net = FCN(num_labels=NUM_OF_CLASSES, batch_size=1)
    net.build_network()
    if LOAD_FILE and LOAD_FILE != "":
        net.load_model_from(LOAD_FILE)
    # incase of uncaught exception or CTRL+C saves the weights and prints stuff
    #  NOTE: DOES NOT include pycharm stop button because that sends a SIGKILL
    atexit.register(lambda x: x.on_exit(), net)
    # runs on every uncaught exception
    sys.excepthook = new_except_hook(net)


    print(get_model_memory_usage(1, net.model))
    print(net.model.summary())

    net.train_keras(TRAINING_DATA_FILE, TRAINING_LABEL_FILE)
    data_gen = load_data_generator(VALIDATION_DATA_FILE, VALIDATION_LABEL_FILE, net.num_labels, shuffle=True, preload=1)
    for image, labels, selection in data_gen:
        # print(image[0].shape)
        # print("printing pool3: " ,net.pool3.predict(image).shape)
        # print("printing pool4: " ,net.pool4.predict(image).shape)
        # print("printing deconv1: " ,net.deconv1.predict(image).shape)
        # print("printing crop1: " ,net.crop1.predict(image).shape)
        # print("printing skip1: " ,net.skip1.predict(image).shape)
        # print("printing deconv2: " ,net.deconv2.predict(image).shape)
        # print("printing crop2: " ,net.crop2.predict(image).shape)
        # print("printing skip2: " ,net.skip2.predict(image).shape)

        pred = net.model.predict(image)
        label = tf.constant(labels[0][0])
        predict = tf.constant(pred[0])
        print(computeIoU(y_true_batch=labels, y_pred_batch=pred, num_labels=net.num_labels))
        plot_images(image[0][0], labels[0][0], pred[0])
    # net.test_network(data_gen, num_of_batches=100)
    # net.save_network()

    print("finished")

if __name__ == '__main__':
    main()