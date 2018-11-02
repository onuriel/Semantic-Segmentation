"""An implementation of the article - 'Fully Convolutional Network for Semantic Segmentation'.
The article can be found: https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_011.pdf"""


# ------------------------------------------- IMPORTS --------------------------------------------------------
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras.applications.imagenet_utils import preprocess_input
from utils.utils import  new_except_hook, get_bilinear_filter, CroppingLike2D, plot_images, evaluate_iou
from utils.metrics import sparse_accuracy_ignoring_last_label
from utils.loss_function import softmax_sparse_crossentropy_ignoring_last_label
from utils.SegDataGenerator import SegDataGenerator
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Conv2DTranspose, Add, ZeroPadding2D
from keras.regularizers import l2
import pickle
from os.path import join
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import atexit
import sys
import argparse


# ------------------------------------------- GLOBALS --------------------------------------------------------


train_file_path = os.path.join('benchmark_RELEASE', 'dataset','train.txt')
val_file_path = os.path.join('benchmark_RELEASE', 'dataset', 'seg11valid.txt')
data_dir = os.path.join('benchmark_RELEASE', 'dataset','img')
label_dir = os.path.join('benchmark_RELEASE','dataset','cls')
classes = 21
TRAINING_LABEL_FILE = label_dir
TRAINING_DATA_FILE = data_dir

PRETRAINED_WEIGHTS_FILE = "vgg16.npy"
MODEL_NAME = "fcn_voc"
WEIGHT_FOLDER = 'weights'
WEIGHT_FILE = "weights"
LOAD_WEIGHT_FILE = None
#LOAD_FILE = WEIGHT_FILE
LOAD_MODEL_FILE = None
MODEL_FOLDER = "models"
OPTOMIZER_FILE = 'optimizer.pkl'



# ------------------------------------------ NETWORK ---------------------------------------------------------


class FCN:

    def __init__(self, num_labels, batch_size=1, weight_decay=1e-4):
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model = None
        self.weight_decay = weight_decay

    def load_weights(self, weight_file=WEIGHT_FILE):
        self.model.load_weights(join(WEIGHT_FOLDER, "{0}.h5".format(weight_file)))

    def build_network(self):
        """
        Creates a fully convolutional network with including upsampling.
        :return: the network
        """
        input_shape = (None, None, 3)
        if self.batch_size is not None:
            image_input = Input(batch_shape=(self.batch_size,)+input_shape)
            padding = ZeroPadding2D(100,  input_shape=(self.batch_size,)+ input_shape)(image_input)
        else:
            image_input = Input(shape=input_shape)
            padding = ZeroPadding2D(100, input_shape=input_shape)(image_input)

        conv1_1 = Conv2D(kernel_size=3, filters=64, activation='relu', name="conv1_1", padding='same', kernel_regularizer=l2(self.weight_decay))(padding)
        conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name="conv1_2", padding='same', kernel_regularizer=l2(self.weight_decay))(conv1_1)
        pool_1 = MaxPool2D(strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_1", padding='same', kernel_regularizer=l2(self.weight_decay))(pool_1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_2", padding='same', kernel_regularizer=l2(self.weight_decay))(conv2_1)
        pool_2 = MaxPool2D(strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_1", padding='same', kernel_regularizer=l2(self.weight_decay))(pool_2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv3_2)
        pool_3 = MaxPool2D(strides=2)(conv3_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_1", padding='same',kernel_regularizer=l2(self.weight_decay))(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)

        conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_1", padding='same',kernel_regularizer=l2(self.weight_decay))(pool_4)
        conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv5_2)
        pool_5 = MaxPool2D(strides=2)(conv5_3)

        # fully conv
        fc6 = Conv2D(filters=4096, kernel_size=7, activation='relu', name="fc6", padding='valid',kernel_regularizer=l2(self.weight_decay))(pool_5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=1, activation='relu', name="fc7", padding='valid',kernel_regularizer=l2(self.weight_decay))(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = Conv2D(filters=self.num_labels, kernel_size=1, name="fc8", padding='same',kernel_regularizer=l2(self.weight_decay))(drop7)

        # up sampling
        deconv1 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv1",kernel_regularizer=l2(self.weight_decay))(score_fr)
        crop_pool4 = CroppingLike2D(deconv1, num_classes=self.num_labels, offset=6)(pool_4)
        skip1 = Conv2D(filters=self.num_labels, kernel_size=1, padding='same', activation=None, name='score_pool4',kernel_regularizer=l2(self.weight_decay))(crop_pool4)
        add_pool_4 = Add()([skip1, deconv1])

        deconv2 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv2",kernel_regularizer=l2(self.weight_decay))(add_pool_4)
        crop_pool3 = CroppingLike2D(deconv2, self.num_labels, offset=8)(pool_3)
        skip2 = Conv2D(filters=self.num_labels, kernel_size=1, activation=None, name="score_pool3", padding="same",kernel_regularizer=l2(self.weight_decay))(crop_pool3)
        add_pool_3 = Add()([skip2, deconv2])

        deconv3 = Conv2DTranspose(filters= self.num_labels,use_bias=False, kernel_size=16, strides=8, activation='linear', name="final",kernel_regularizer=l2(self.weight_decay))(add_pool_3)
        output = CroppingLike2D(image_input, self.num_labels, offset=12)(deconv3)
        self.model = Model(inputs=image_input, outputs=output)

        # load and compile model
        if LOAD_WEIGHT_FILE is None and LOAD_MODEL_FILE is None:
            self.set_keras_weights()

        target = tf.placeholder(dtype='int32', shape=(None, None, None, None))
        self.model.compile(optimizer='adam', loss=softmax_sparse_crossentropy_ignoring_last_label, metrics=[sparse_accuracy_ignoring_last_label], target_tensors=[target])
        return output

    def set_keras_weights(self):
        """
        This function loads the starting weights of the network.
        """
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
                if layer.name == 'final':
                    layer.set_weights([bilinear_weights])
                    continue
                layer.set_weights([bilinear_weights, current_weights[1]])

    def train(self):
        """
        This function trains the network.
        """
        checkpoint = ModelCheckpoint(join(WEIGHT_FOLDER, "checkpoint_weights_batch_1.h5"), monitor='val_sparse_accuracy_ignoring_last_label', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
        callback_list = [checkpoint]
        current_dir = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(current_dir, MODEL_FOLDER+"/" + MODEL_NAME)
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        def get_file_len(file_path):
            fp = open(file_path)
            lines = fp.readlines()
            fp.close()
            return len(lines)

        # from Keras documentation: Total number of steps (batches of samples) to yield from generator before
        # declaring one epoch finished and starting the next epoch. It should typically be equal to the number
        # of unique samples of your dataset divided by the batch size.
        steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(self.batch_size)))
        target_size = (320, 320)
        train_datagen = SegDataGenerator(
                                         zoom_range=[0.5, 2.0],
                                         zoom_maintain_shape=True,
                                         crop_mode='random',
                                         crop_size=target_size,
                                         rotation_range=0.,
                                         shear_range=0,
                                         horizontal_flip=True,
                                         fill_mode='constant',
                                         label_cval=255,
                                         preprocess_input=True)

        val_datagen = SegDataGenerator(preprocess_input=True)

        history = self.model.fit_generator(generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=".jpg",
            label_dir=label_dir, label_suffix=".png",
            classes=classes,
            target_size=target_size,
            color_mode='rgb',
            batch_size=self.batch_size, shuffle=True,
            ignore_label=255),
            epochs=50,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=1,
            validation_data=val_datagen.flow_from_directory(
                file_path=val_file_path,
                data_dir=data_dir, data_suffix=".jpg",
                label_dir=label_dir, label_suffix=".png",
                classes=classes,
                target_size=target_size, color_mode='rgb',
                batch_size=self.batch_size, shuffle=True,),
            validation_steps=40)
        
        with open('history_batch_size1', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        self.save_weights_and_model("final")

    def save_weights_and_model(self, prefix=None):
        """
        This function saves the weights and model.
        :param prefix: The prefix of the files saved.
        """
        model_name = MODEL_NAME
        weight_name = WEIGHT_FILE
        if prefix is not None:
            model_name = "{0}_{1}".format(prefix, model_name)
            weight_name = "{0}_{1}".format(prefix, weight_name)

        self.model.save(join(MODEL_FOLDER, "{0}.h5".format(model_name)))
        self.model.save_weights(join(WEIGHT_FOLDER, "{0}.h5".format(weight_name)))

    def save_network(self, file_prefix=None):
        """
        This function saves the network.
        :param file_prefix: The prefix of files saved.
        """
        self.save_weights_and_model(file_prefix)
        print("model Saved")

    def on_exit(self):
        """
        This function is for safety reasons. When exiting the program before it finishes running it will offer
        to save the networks progress.
        """
        model_path = 'auto_exit'
        a = input("save model to {0} ? (y/n) : \n".format(model_path))
        if a == 'y':
            self.save_network(model_path)
            print("weights saved to : {0}.h5".format(model_path))

    def test_network(self, test_path, test_data_dir, test_label_dir):
        """
        This function tests the network.
        :param test_path: The path to the test data.
        :param test_data_dir: The directory in the path for the test data.
        :param test_label_dir: The directory in the path for the labels of the test data.
        """
        agg_tp, agg_fn, agg_fp = 0, 0, 0

        def get_file_len(file_path):
            fp = open(file_path)
            lines = fp.readlines()
            fp.close()
            return len(lines)

        size = get_file_len(test_path)
        datagen = SegDataGenerator()
        data_iter = datagen.flow_from_directory(file_path=test_path,
                                                data_dir=test_data_dir, data_suffix='.jpg',
                                                label_dir=test_label_dir, label_suffix='.png',
                                                classes=classes, color_mode='rgb', batch_size=1)
        for index in range(size):
            data, label = data_iter._get_batches_of_transformed_samples([index])
            predict = self.model.predict(data)
            prediction = np.argmax(predict.reshape(predict.shape[1:]), axis=2).astype(np.int)
            label = label.reshape(label.shape[1:3])
            tp, fn, fp = evaluate_iou(label, prediction)
            if tp + fp + fn == 0:
                iou = 1.
            else:
                iou = tp / float(tp + fp + fn)
            print("Current IoU : {}".format(iou))
            agg_tp += tp
            agg_fn += fn
            agg_fp += fp
            if agg_tp + agg_fp + agg_fn == 0:
                iou = 1.
            else:
                iou = agg_tp / float(agg_tp + agg_fp + agg_fn)

            print("aggregated IoU : {}".format(iou))

        if agg_tp + agg_fp + agg_fn == 0:
            iou = 1.
        else:
            iou = agg_tp / float(agg_tp + agg_fp + agg_fn)

        print("total IoU : {}".format(iou))


# ------------------------------------------ MAIN ---------------------------------------------------------


def main(train, batch_size):

    net = FCN(num_labels=classes, batch_size=batch_size)
    if train:
        atexit.register(lambda x: x.on_exit(), net)
        sys.excepthook = new_except_hook(net)

        if LOAD_MODEL_FILE is not None and LOAD_MODEL_FILE != "":
            net.model = load_model(join(MODEL_FOLDER,LOAD_MODEL_FILE), custom_objects={"CroppingLike2D": CroppingLike2D})
            print("loaded existing model")
        else:
            net.build_network()
            if LOAD_WEIGHT_FILE and LOAD_WEIGHT_FILE != "":
                net.load_weights(LOAD_WEIGHT_FILE)
                print("loaded weights and optimizer weights")

        net.train()

        # Test results
        net.test_network(val_file_path, data_dir, label_dir)

        # Display results
        # TODO check if this is the most updated way.
        def get_file_len(file_path):
            fp = open(file_path)
            lines = fp.readlines()
            fp.close()
            return len(lines)

        size = get_file_len(val_file_path)
        datagen = SegDataGenerator()
        data_iter = datagen.flow_from_directory(file_path=val_file_path,
                                                data_dir=data_dir, data_suffix='.jpg',
                                                label_dir=label_dir, label_suffix='.png',
                                                classes=classes, color_mode='rgb', batch_size=1)
        for i in range(5):
            index = np.random.choice(size)
            data, label = data_iter._get_batches_of_transformed_samples([index])

            predict = net.model.predict(preprocess_input(data.copy()))
            prediction = np.argmax(predict.reshape(predict.shape[1:]), axis=2).astype(np.int)
            plot_images(data.squeeze(), label.squeeze(), prediction)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--training', '-training', type=bool, default=False, help='Run training')
    p.add_argument('--batch_size', '-batch_size', type=int, default=1, help='Batch size during training')
    args = p.parse_args()
    main(args.training, args.batch_size)
