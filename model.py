"""An implementation of the article - 'Fully Convolutional Network for Semantic Segmentation'.
The article can be found: https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_011.pdf"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# to suppress tensorflow messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1
set_session(tf.Session(config=config))



from utils.utils import load_data_generator, new_except_hook, get_bilinear_filter, CroppingLike2D, plot_images, evaluate_iou
from utils.metrics import  sparse_accuracy_ignoring_last_label
from utils.loss_function import softmax_sparse_crossentropy_ignoring_last_label, sparse_cross_entropy
from utils.SegDataGenerator import SegDataGenerator
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Conv2DTranspose, Add, Activation
from keras.regularizers import l2
from keras.optimizers import SGD
import pickle
from os.path import join
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
import atexit
import sys

train_file_path = 'benchmark_RELEASE/dataset/train.txt'  # Data/VOClarge/VOC2012/ImageSets/Segmentation
val_file_path = 'benchmark_RELEASE/dataset/seg11valid.txt'
data_dir = 'benchmark_RELEASE/dataset/img'
label_dir = 'benchmark_RELEASE/dataset/cls'
classes = 21
TRAINING_LABEL_FILE = label_dir
TRAINING_DATA_FILE = data_dir
# VALIDATION_LABEL_FILE = "ADEChallengeData2016/annotations/validation"
# VALIDATION_DATA_FILE = "ADEChallengeData2016/images/validation"
PRETRAINED_WEIGHTS_FILE = "vgg16.npy"
MODEL_NAME = "fcn_voc"
WEIGHT_FOLDER = 'weights'
WEIGHT_FILE = "weights"
LOAD_WEIGHT_FILE = "final_weights"
#LOAD_FILE = WEIGHT_FILE
LOAD_MODEL_FILE = None
MODEL_FOLDER = "models"
OPTOMIZER_FILE = 'optimizer.pkl'

NUM_OF_CLASSES = 21



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
        else:
            image_input = Input(shape=input_shape)
       # identity = Activation('linear')(image_input)
        #self.identity = Model(image_input, output=identity)

        conv1_1 = Conv2D(kernel_size=3, filters=64, activation='relu', name="conv1_1", padding='same', kernel_regularizer=l2(self.weight_decay))(image_input)
        conv1_2 = Conv2D(filters=64, kernel_size=3, activation='relu', name="conv1_2", padding='same', kernel_regularizer=l2(self.weight_decay))(conv1_1)
        pool_1 = MaxPool2D(strides=2)(conv1_2)

        conv2_1 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_1", padding='same', kernel_regularizer=l2(self.weight_decay))(pool_1)
        conv2_2 = Conv2D(filters=128, kernel_size=3, activation='relu', name="conv2_2", padding='same')(conv2_1)
        pool_2 = MaxPool2D(strides=2)(conv2_2)

        conv3_1 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_1", padding='same', kernel_regularizer=l2(self.weight_decay))(pool_2)
        conv3_2 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv3_1)
        conv3_3 = Conv2D(filters=256, kernel_size=3, activation='relu', name="conv3_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv3_2)
        pool_3 = MaxPool2D(strides=2)(conv3_3)
        # self.pool3 = Model(input=image_input, output=pool_3)

        conv4_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_1", padding='same',kernel_regularizer=l2(self.weight_decay))(pool_3)
        conv4_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv4_1)
        conv4_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv4_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv4_2)
        pool_4 = MaxPool2D(strides=2)(conv4_3)
        # self.pool4 = Model(input=image_input, output=pool_4)

        conv5_1 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_1", padding='same',kernel_regularizer=l2(self.weight_decay))(pool_4)
        conv5_2 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_2", padding='same',kernel_regularizer=l2(self.weight_decay))(conv5_1)
        conv5_3 = Conv2D(filters=512, kernel_size=3, activation='relu', name="conv5_3", padding='same',kernel_regularizer=l2(self.weight_decay))(conv5_2)
        pool_5 = MaxPool2D(strides=2)(conv5_3)

        #fully conv
        fc6 = Conv2D(filters=4096, kernel_size=7, activation='relu', name="fc6", padding='same',kernel_regularizer=l2(self.weight_decay))(pool_5)
        drop6 = Dropout(0.5)(fc6)
        fc7 = Conv2D(filters=4096, kernel_size=1, activation='relu', name="fc7", padding='same',kernel_regularizer=l2(self.weight_decay))(drop6)
        drop7 = Dropout(0.5)(fc7)
        score_fr = Conv2D(filters=self.num_labels, kernel_size=1, name="fc8", padding='same',kernel_regularizer=l2(self.weight_decay))(drop7)
        # self.score_fr = Model(input=image_input, output=score_fr)
        deconv1 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv1",kernel_regularizer=l2(self.weight_decay))(score_fr) #deconv 32
       # self.deconv1 = Model(input=image_input, output=deconv1)
        crop_deconv1 = CroppingLike2D(pool_4, num_classes=self.num_labels)(deconv1)
       # self.crop1 = Model(input=image_input, output=crop_deconv1)
        skip1 = Conv2D(filters=self.num_labels, kernel_size=1, padding='same', activation=None, name='score_pool4',kernel_regularizer=l2(self.weight_decay))(pool_4)
        # self.skip1 = Model(input=image_input, output=skip1)
        add_pool_4 = Add()([skip1, crop_deconv1])

        # self.add_pool4 = Model(input=image_input, output=add_pool_4)
        deconv2 = Conv2DTranspose(filters=self.num_labels, kernel_size=4, strides=2, activation=None, name="deconv2",kernel_regularizer=l2(self.weight_decay))(add_pool_4)
        #self.deconv2 = Model(input=image_input, output=deconv2)
        crop_deconv2 = CroppingLike2D(pool_3, self.num_labels)(deconv2)
        #self.crop2 = Model(input=image_input, output=crop_deconv2)

        skip2 = Conv2D(filters=self.num_labels, kernel_size=1, activation=None, name="score_pool3", padding="same",kernel_regularizer=l2(self.weight_decay))(pool_3)
        # self.skip2 = Model(input=image_input, output=skip2)
        add_pool_3 = Add()([skip2, crop_deconv2])

        deconv3 = Conv2DTranspose(filters= self.num_labels,use_bias=False, kernel_size=16, strides=8, activation='linear', name="final",kernel_regularizer=l2(self.weight_decay))(add_pool_3)
        output = CroppingLike2D(image_input, self.num_labels)(deconv3)
        self.model = Model(inputs=image_input, outputs=output)
        optimizer = SGD(lr=10e-4, momentum=0.99, nesterov=True)
        if LOAD_WEIGHT_FILE is None and LOAD_MODEL_FILE is None:
            self.set_keras_weights()

        target = tf.placeholder(dtype='int32', shape=(None, None, None, None))
        self.model.compile(optimizer='adam', loss=softmax_sparse_crossentropy_ignoring_last_label, metrics=[sparse_accuracy_ignoring_last_label], target_tensors=[target])
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
                if layer.name == 'final':
                    layer.set_weights([bilinear_weights])
                    continue
                layer.set_weights([bilinear_weights, current_weights[1]])



    def train(self, data_file, label_file, test_data_file=None, test_label_file=None):
        checkpoint = ModelCheckpoint(join(WEIGHT_FOLDER, "checkpoint_weights_batch_1.h5"), monitor='val_sparse_accuracy_ignoring_last_label', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
        # checkpoint = LambdaCallback(on_epoch_end= lambda epoch, log: self.save_weights_and_opt_state())
        # csvlogger = CSVLogger('training.log', append=True)
        callback_list = [checkpoint]
        current_dir = os.path.dirname(os.path.realpath(__file__))
        save_path = os.path.join(current_dir, MODEL_FOLDER+"/" +MODEL_NAME )
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)

        def get_file_len(file_path):
            fp = open(file_path)
            lines = fp.readlines()
            fp.close()
            return len(lines)

        # from Keras documentation: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished
        # and starting the next epoch. It should typically be equal to the number of unique samples of your dataset divided by the batch size.
        steps_per_epoch = int(np.ceil(get_file_len(train_file_path) / float(self.batch_size)))
        target_size=(360,360)
        train_datagen = SegDataGenerator(
                                         zoom_range=[0.5, 2.0],
                                         zoom_maintain_shape=True,
                                         crop_mode='random',
                                         crop_size=target_size,
                   #                      pad_size=(505, 505),
                                         rotation_range=0.,
                                         shear_range=0,
                                         horizontal_flip=True,
                                         #channel_shift_range=20.,
                                         fill_mode='constant',
                                         label_cval=255
                   )

        val_datagen = SegDataGenerator()


        history = self.model.fit_generator(generator=train_datagen.flow_from_directory(
            file_path=train_file_path,
            data_dir=data_dir, data_suffix=".jpg",
            label_dir=label_dir, label_suffix=".png",
            classes=classes,
            target_size=target_size,
            color_mode='rgb',
            batch_size=self.batch_size, shuffle=True,
            ignore_label=255),
            epochs=20,
            steps_per_epoch=steps_per_epoch,
            callbacks=callback_list,
            verbose=1,
            validation_data=val_datagen.flow_from_directory(
                file_path=val_file_path,
                data_dir=data_dir, data_suffix=".jpg",
                label_dir=label_dir, label_suffix=".png",
                classes=classes,
                target_size=target_size, color_mode='rgb',
                batch_size=self.batch_size, shuffle=True),
            validation_steps=40)
        
        with open('history_batch_size2', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        self.save_weights_and_model("final")


    def save_weights_and_model(self, prefix=None):
        model_name = MODEL_NAME
        weight_name = WEIGHT_FILE
        if prefix is not None:
            model_name = "{0}_{1}".format(prefix, model_name)
            weight_name = "{0}_{1}".format(prefix, weight_name)

        self.model.save(join(MODEL_FOLDER, "{0}.h5".format(model_name)))
        self.model.save_weights(join(WEIGHT_FOLDER, "{0}.h5".format(weight_name)))



    def save_network(self, file_prefix):
        self.save_weights_and_model(file_prefix)
        print("model Saved")

    def on_exit(self):
        model_path = 'auto_exit'
        a = input("save model to {0} ? (y/n) : \n".format(model_path))
        if a == 'y':
            self.save_network(model_path)
            print("weights saved to : {0}.h5".format(model_path))

    def test_network(self, test_path, test_data_dir, test_label_dir):
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



def main():

    net = FCN(num_labels=NUM_OF_CLASSES, batch_size=40)
    if LOAD_MODEL_FILE is not None and LOAD_MODEL_FILE != "":
        net.model = load_model(join(MODEL_FOLDER,LOAD_MODEL_FILE), custom_objects={"CroppingLike2D": CroppingLike2D})
        print("loaded existing model")
    else:
        net.build_network()
        if LOAD_WEIGHT_FILE and LOAD_WEIGHT_FILE != "":
            net.load_weights(LOAD_WEIGHT_FILE)
            print("loaded weights and optimizer weights")

    # incase of uncaught exception or CTRL+C saves the weights and prints stuff
    #  NOTE: DOES NOT include pycharm stop button because that sends a SIGKILL
    atexit.register(lambda x: x.on_exit(), net)
    # runs on every uncaught exception
    sys.excepthook = new_except_hook(net)
    net.train(TRAINING_DATA_FILE, TRAINING_LABEL_FILE)
    #data_gen = load_data_generator(TRAINING_DATA_FILE, TRAINING_LABEL_FILE, net.num_labels, shuffle=True, preload=1)
    net.test_network(val_file_path, data_dir, label_dir)
    #for image, labels, selection in data_gen:
    #        pred = net.model.predict(image[0].reshape((1,)+image[0].shape))
    #        import pdb
           # pdb.set_trace()
    #        label = tf.constant(labels[0])
    #        predict = tf.constant(pred)
    #        print(softmax_sparse_crossentropy_ignoring_last_label(label, predict).eval(session=tf.Session()))
    # data_gen = load_data_generator(TRAINING_DATA_FILE, TRAINING_LABEL_FILE, net.num_labels, shuffle=True, preload=1, )
    # for image, labels, selection in data_gen:
    #     pred = net.model.predict_on_batch(image)
    #     plot_images(image[0][0], labels[0][0], pred)
    # net.test_network(data_gen, num_of_batches=100)
    # net.save_network()

    print("finished")








if __name__ == '__main__':
    main()
