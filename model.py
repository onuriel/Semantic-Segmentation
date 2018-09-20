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

# from keras.preprocessing.image import ImageDataGenerator

from utils import load_data_generator, new_except_hook, get_bilinear_filter, CroppingLike2D, plot_images, DataGen
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Conv2DTranspose, Activation, Add
from os.path import join
from keras.models import Model, load_model
from keras.callbacks import CSVLogger
import numpy as np
import atexit
import sys


TRAINING_LABEL_FILE = "VOC2012/SegmentationClass"
TRAINING_DATA_FILE = "VOC2012/SegmentationData"
VALIDATION_LABEL_FILE = "ADEChallengeData2016/annotations/validation"
VALIDATION_DATA_FILE = "ADEChallengeData2016/images/validation"
PRETRAINED_WEIGHTS_FILE = "vgg16.npy"
WEIGHT_FOLDER = 'weights'
WEIGHT_FILE = "weights"
LOAD_WEIGHT_FILE = None
#LOAD_FILE = WEIGHT_FILE
LOAD_MODEL_FILE = None
MODEL_FOLDER = "models"
OPTOMIZER_FILE = 'optimizer.pkl'

NUM_OF_CLASSES = 21
SAMPLES_PER_EPOCH = 20210



class FCN:

    def __init__(self, num_labels, batch_size=1):
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.model = None


    def load_weights_and_opt_state(self, weight_file=WEIGHT_FILE):
        # load json and create model
        # json_file = open('model.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(join(WEIGHT_FOLDER, "{0}.h5".format(weight_file)))
        # self.model._make_train_function()
        # with open(join(WEIGHT_FOLDER,"{0}_{1}".format(weight_file, OPTOMIZER_FILE)), 'rb') as f:
        #     weight_values = pickle.load(f)
        # self.model.optimizer.set_weights(weight_values)
        # print("Loaded weights from disk")



    def build_network(self):
        """
        Creates a fully convolutional network with including upsampling.
        :return: the network
        """

        image_input = Input(shape=(None, None, 3))

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
        # self.pool3 = Model(input=image_input, output=pool_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_1", padding='same')(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_2", padding='same')(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_3", padding='same')(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)
        # self.pool4 = Model(input=image_input, output=pool_4)

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
        # self.score_fr = Model(input=image_input, output=score_fr)
        deconv1 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv1")(score_fr) #deconv 32
        # self.deconv1 = Model(input=image_input, output=deconv1)
        crop_deconv1 = CroppingLike2D(pool_4, num_classes=self.num_labels)(deconv1)
        # self.crop1 = Model(input=image_input, output=crop_deconv1)
        skip1 = Conv2D(filters=self.num_labels, kernel_size=1, padding='same', activation=None, name='score_pool4')(pool_4)
        # self.skip1 = Model(input=image_input, output=skip1)
        add_pool_4 = Add()([skip1, crop_deconv1])

        # self.add_pool4 = Model(input=image_input, output=add_pool_4)
        deconv2 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv2")(add_pool_4)
        # self.deconv2 = Model(input=image_input, output=deconv2)
        crop_deconv2 = CroppingLike2D(pool_3, self.num_labels)(deconv2)
        # self.crop2 = Model(input=image_input, output=crop_deconv2)

        skip2 = Conv2D(filters=self.num_labels, kernel_size=1, activation=None, name="score_pool3", padding="same")(pool_3)
        # self.skip2 = Model(input=image_input, output=skip2)
        add_pool_3 = Add()([skip2, crop_deconv2])

        deconv3 = Conv2DTranspose(filters= self.num_labels, kernel_size=16, strides=8, activation=None, name="final")(add_pool_3)
        crop_deconv3 = CroppingLike2D(image_input, self.num_labels)(deconv3)


        output = Activation('softmax', name="softmax_layer")(crop_deconv3)
        self.model = Model(inputs=image_input, outputs=output)
        if LOAD_WEIGHT_FILE is None and LOAD_MODEL_FILE is None:
            self.set_keras_weights()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
        # data_generator = load_data_generator(data_file, label_file, num_classes=self.num_labels, preload=5, batch_size=self.batch_size, shuffle=True, return_with_selection=False)
        test_generator = load_data_generator(test_data_file, test_label_file, num_classes=self.num_labels, preload=5, batch_size=self.batch_size, shuffle=True, return_with_selection=False)
        # checkpoint = ModelCheckpoint(join(WEIGHT_FOLDER, "weights.h5"), monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
        # checkpoint = LambdaCallback(on_epoch_end= lambda epoch, log: self.save_weights_and_opt_state())
        # csvlogger = CSVLogger('training.log', append=True)
        # callback_list = [csvlogger]
        data_generator = DataGen(data_file, label_file, num_classes=self.num_labels, preload=5, batch_size=self.batch_size, shuffle=True, return_with_selection=False, split=0.2)
        steps = 0
        num_epochs = 1
        while True:

            for image, label in data_generator:
                try:
                    loss, acc = self.model.train_on_batch(image, label)
                except Exception:
                    continue
                steps += 1
                if steps%2 and steps != 0:
                    num_epochs += 1
                    if data_generator.split is not None:
                        data_generator.validate = True
                        test_generator = data_generator
                    val_loss, val_acc = self.model.evaluate_generator(test_generator, steps=20, max_queue_size=5, verbose=1)
                    data_generator.validate = False
                    print("val_loss {0}, val_acc {1}".format(val_loss, val_acc))
                    self.save_weights_and_opt_state()
                print("Finished {0}/2000 steps in epoch {1} ".format(steps, num_epochs))
                print("loss {0}, acc {1}".format(loss, acc))






        # history = self.model.fit_generator(data_generator, max_queue_size=4, steps_per_epoch=2000, validation_data=test_generator, verbose=1, validation_steps=20, epochs=50, callbacks=callback_list)
        # self.save_network("final_model")
        # with open('trainHistoryDict', 'wb') as file_pi:
        #     pickle.dump(history.history, file_pi)

    def save_weights_and_opt_state(self, prefix=None):
        model_name = "model"
        weight_name = "weights"
        if prefix is not None:
            model_name = "{0}_{1}".format(prefix, model_name)
            weight_name = "{0}_{1}".format(prefix, weight_name)

        self.model.save(join(MODEL_FOLDER, "{0}.h5".format(model_name)))
        self.model.save_weights(join(WEIGHT_FOLDER, "{0}.h5".format(weight_name)))
        # symbolic_weights = getattr(self.model.optimizer, 'weights')
        # weight_values = K.batch_get_value(symbolic_weights)
        # with open(join(WEIGHT_FOLDER, "{0}_{1}".format(weight_name,OPTOMIZER_FILE)), 'wb') as f:
        #     pickle.dump(weight_values, f)


    def save_network(self, file_prefix):
        self.save_weights_and_opt_state(file_prefix)
        print("model Saved")

    def on_exit(self):
        model_path = 'auto_exit'
        a = input("save model to {0} ? (y/n) : \n".format(model_path))
        if a == 'y':
            self.save_network(model_path)
            print("weights saved to : {0}.h5".format(model_path))

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
    if LOAD_MODEL_FILE is not None and LOAD_MODEL_FILE != "":
        net.model = load_model(join(MODEL_FOLDER,LOAD_MODEL_FILE), custom_objects={"CroppingLike2D": CroppingLike2D})
        print("loaded existing model")
    else:
        net.build_network()
        if LOAD_WEIGHT_FILE and LOAD_WEIGHT_FILE != "":
            net.load_weights_and_opt_state(LOAD_WEIGHT_FILE)
            print("loaded weights and optimizer weights")
                            

    # incase of uncaught exception or CTRL+C saves the weights and prints stuff
    #  NOTE: DOES NOT include pycharm stop button because that sends a SIGKILL
    atexit.register(lambda x: x.on_exit(), net)
    # runs on every uncaught exception
    sys.excepthook = new_except_hook(net)

    # net.train_keras(TRAINING_DATA_FILE, TRAINING_LABEL_FILE)
    data_gen = load_data_generator(TRAINING_DATA_FILE, TRAINING_LABEL_FILE, net.num_labels, shuffle=True, preload=1, )
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

        pred = net.model.predict_on_batch(image)
        label = tf.constant(labels[0][0])
        predict = tf.constant(pred[0])
        # print(computeIoU(y_true_batch=labels, y_pred_batch=pred, num_labels=net.num_labels))
        plot_images(image[0][0], labels[0][0], pred)
    # net.test_network(data_gen, num_of_batches=100)
    # net.save_network()

    print("finished")

if __name__ == '__main__':
    main()
