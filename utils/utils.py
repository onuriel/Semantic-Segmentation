import numpy as np
import sys
from cv2 import imread
from os import getpid
from os import listdir
from os.path import join
from keras.utils import to_categorical
from PIL import Image
import tensorflow as tf
import keras.backend as K
from keras.layers import Lambda, Cropping2D
from keras.engine import InputSpec, Layer
SYSTEM_EXCEPT_HOOK = sys.excepthook
import matplotlib.pyplot as plt
import psutil
import shutil
MEGA = 10 ** 6
MEGA_STR = ' ' * MEGA




class CroppingLike2D(Layer):
    def __init__(self, target=None, num_classes=151, **kwargs):
        """Crop to target.
        If only one `offset` is set, then all dimensions are offset by this amount.
        """
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = "channels_last"
        self.target_shape = (None, None, None, num_classes)
        self.target = K.zeros(shape=(1,1,1,1))  if target is None else target
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                self.target_shape[1],
                self.target_shape[2],
                input_shape[3])
    def call(self, inputs, **kwargs):
        orig_shape = K.shape(self.target)
        input_image = inputs
        
        return tf.image.crop_to_bounding_box(input_image, 0, 0, orig_shape[1], orig_shape[2])
                                                                                                                        

def crop2d(orig_image):
    # deconv_shape = K.shape(deconv_image)
    orig_shape = K.shape(orig_image)
    # extra_h = deconv_shape[1] - orig_shape[1]
    # extra_w = deconv_shape[2] - orig_shape[2]
    crop = Lambda(lambda x: tf.image.crop_to_bounding_box(x, 0, 0, orig_shape[1], orig_shape[2]))
    return crop


def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def computeIoU(y_pred_batch, y_true_batch, num_labels):
    return np.mean(np.asarray([pixelAccuracy(y_pred_batch[i], y_true_batch[i], num_labels) for i in range(len(y_true_batch))]))

def print_memory_usage():
    """Prints current memory usage stats.
    See: https://stackoverflow.com/a/15495136
    :return: None
     """
    PROCESS = psutil.Process(getpid())
    total, available, percent, used, free = psutil.virtual_memory()[:5]
    total, available, used, free = total / MEGA, available / MEGA, used / MEGA, free / MEGA
    proc = PROCESS.memory_info()[1] / MEGA
    print('process = %s total = %s available = %s used = %s free = %s percent = %s' % (proc, total, available, used, free, percent))


def pixelAccuracy(y_pred, y_true, num_classes):
    shape = np.shape(y_pred)
    img_rows = shape[0]
    img_cols = shape[1]
    y_pred = np.argmax(np.reshape(y_pred,[num_classes,img_rows,img_cols]),axis=0)
    if np.max(y_pred) == 0:
        print("All Zeros on {0}")
    y_true = np.argmax(np.reshape(y_true,[num_classes,img_rows,img_cols]),axis=0)
    y_pred = y_pred * (y_true>0)

    return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)

def get_listdir_cache():
    folder_dict = {}
    def get_list(folder):
        if folder not in folder_dict:
            folder_dict[folder] = sorted(listdir(folder))
        return folder_dict[folder]
    return get_list

LIST_DIR_CACHE = get_listdir_cache()


def load_data(sample_file, label_file, num_classes):
    data = load_images(sample_file)
    labels = load_images(label_file)
    preprocess_data(data, labels, num_classes)
    for i, label in enumerate(labels):
        height, width, channels = labels[i].shape
        data[i] = data[i].reshape(1,height, width, 3)
        data[i] = data[i].astype(np.float32)
        labels[i] = np.reshape(to_categorical(label[:,:,0], num_classes), (1,height,width,num_classes))
    return data, labels


class DataGen:
    def __init__(self, data_folder, label_folder,num_classes=151, preload=10, batch_size=1, cursor=0, shuffle=False, return_with_selection=True, split=None):
        self.data_foler = data_folder
        self.label_folder = label_folder
        self.data_folder_dir = np.array(sorted(listdir(data_folder)))
        self.label_folder_dir = np.array(sorted(listdir(label_folder)))
        self.data_size = len(self.data_folder_dir)
        self.num_classes = num_classes
        self.preload = preload
        self.batch_size = batch_size
        self.cursor = cursor
        self.shuffle = shuffle
        self.return_with_selection = return_with_selection
        self.split = split
        self.validate = False
        if self.split is not None:
            self.validation_selection = np.random.choice(self.data_size, int(self.data_size*self.split))
            self.train_selection = np.delete(self.data_size, self.validation_selection)



    def __iter__(self):
        return self

    def __next__(self):
        data = [0] * self.preload
        selection = None
        files_list_data = self.data_folder_dir
        files_list_label = self.label_folder_dir
        while len(data) > 0:
            print(self.cursor)
            if self.shuffle:
                if self.split is None:
                    selection = np.random.choice(self.data_size, self.preload)
                else:
                    if self.validate:
                        selection = np.random.choice(len(self.validation_selection), self.preload)
                        files_list_data = self.label_folder_dir[self.validation_selection]
                        files_list_label = self.label_folder_dir[self.validation_selection]
                    else:
                        selection = np.random.choice(len(self.train_selection), self.preload)
                        files_list_label = self.label_folder_dir[self.train_selection]
                        files_list_data = self.data_folder_dir[self.train_selection]
            data = load_images(self.data_foler, self.preload, cursor=self.cursor, selection=selection,list_of_files=files_list_data)
            labels = load_images(self.label_folder, self.preload, cursor=self.cursor, selection=selection, list_of_files=files_list_label)
            preprocess_data(data, labels, self.num_classes)
            for i in range(0, len(labels), self.batch_size):
                if self.return_with_selection:
                    yield data[i:i + self.batch_size], labels[i:i + self.batch_size], selection[i:i + self.batch_size]
                else:
                    print(selection[i:i + self.batch_size])
                    yield data[i:i + self.batch_size], labels[i:i + self.batch_size]
            self.cursor += self.preload
        return




def load_images(folder, num_of_images=20, cursor=0, selection=None, list_of_files=None):
    dataset = []
    if list_of_files is None:
        list_of_files = LIST_DIR_CACHE(folder)
    if selection is not None:
        for i in selection:
            #img = imread(join(folder, list_of_files[i]))
            img = np.array(Image.open(join(folder, list_of_files[i])))
            print(list_of_files[i])
            dataset.append(img)
        return dataset

    for _file in list_of_files[cursor:cursor + num_of_images]:
        img = imread(folder + "/" + _file)
        print(_file)
        dataset.append(img)
    return dataset

def load_data_generator(sample_file, label_file, num_classes=151, preload=10, batch_size=1, cursor=0, shuffle=False, return_with_selection=True):
    data = [0]*preload
    selection = None
    while len(data) > 0:
        print(cursor)
        if shuffle:
            selection = np.random.choice(len(LIST_DIR_CACHE(sample_file)), preload)
        data = load_images(sample_file, preload, cursor=cursor, selection=selection)
        labels = load_images(label_file, preload, cursor=cursor, selection=selection)
        #preprocess_data(data, labels, num_classes)
        for i in range(0,len(labels),batch_size):
            if return_with_selection:
                yield data[i:i+batch_size], labels[i:i+batch_size], selection[i:i+batch_size]
            else:
                print(selection[i:i+batch_size])
                yield data[i:i+batch_size], labels[i:i+batch_size]
        cursor += preload
    return


def preprocess_data(data, labels, num_classes):
    for i, label in enumerate(labels):
        height, width, channels = labels[i].shape
        data[i] = data[i].reshape(1, height, width, 3)
        data[i] = data[i].astype(np.float32)
        labels[i] = np.reshape(to_categorical(label[:, :, 0], num_classes), (1, height, width, num_classes))


def new_except_hook(network):
    def excepthook(*args, **kwargs):
        print("in except hook")
        network.on_exit()
        SYSTEM_EXCEPT_HOOK(*args, **kwargs)
    return excepthook


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

    return weights
    # init = tf.constant_initializer(value=weights,
    #                                dtype=tf.float32)

    # bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
    #                                    shape=weights.shape)
    # return bilinear_weights


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def upsample_layer(bottom, n_channels, name, upscale_factor):
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

def transfer_only_segmentation_image_files(label_folder, data_folder, new_folder_name='Data'):
    label_files = LIST_DIR_CACHE(label_folder)
    for file in label_files:
        file_name = file.split('.')[0]
        new_file_name = file_name+'.jpg'
        data_file = join(data_folder, new_file_name)
        shutil.copy2(data_file, new_folder_name)


def complete_seg11valid_set(source_folder, destination_folder, valid_txt_file, suffix):
    with open(valid_txt_file, 'r') as valid_file:
        list_of_files = set(valid_file.read().splitlines())
        dest_files = listdir(destination_folder)
        src_files = listdir(source_folder)
        for filename in list_of_files:
            if filename+suffix not in dest_files and filename+suffix in src_files:
                shutil.copy2(join(source_folder, filename+suffix), destination_folder)


@as_keras_metric
def mean_iou( y_true, y_pred, num_classes):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

def create_mean_iou(y_true, y_pred, num_classes):
    def my_mean_iou(y_true, y_pred):
        return computeIoU(y_pred, y_true, num_classes)
    return my_mean_iou


def evaluate_iou(gt_labels, pred_labels):

    correct_predictions = gt_labels == pred_labels
    positive_predictions = pred_labels != 0

    # correct, positive prediction -> True positive
    tp = np.sum(correct_predictions & positive_predictions)

    # incorrect, negative prediction (using De Morgan's law) -> False negative
    fn = np.sum(np.logical_not(correct_predictions | positive_predictions))

    # incorrect, positive prediction -> False positive
    fp = np.sum(np.logical_not(correct_predictions) & positive_predictions)

    return tp, fn, fp

def plot_images(image, label, prediction):
    label = np.argmax(label, axis=2)
    prediction = np.argmax(prediction, axis=2)
    print(np.mean(prediction == label))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(label)
    plt.title("True")
    plt.subplot(1,3,3)
    plt.imshow(prediction)
    plt.title("Prediction")
    plt.show()


if __name__=='__main__':
    complete_seg11valid_set("VOCdevkit/VOC2012/JPEGImages",'benchmark_RELEASE/dataset/img/','benchmark_RELEASE/dataset/seg11valid.txt', '.jpg')
    complete_seg11valid_set("VOCdevkit/VOC2012/SegmentationClass",'benchmark_RELEASE/dataset/cls/','benchmark_RELEASE/dataset/seg11valid.txt', '.png')
    #transfer_only_segmentation_image_files("../VOCdevkit/VOC2012/SegmentationClass","../VOCdevkit/VOC2012/JPEGImages", join('../VOCdevkit/VOC2012', "SegmentationData"))
