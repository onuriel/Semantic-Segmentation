import numpy as np
import sys
from os import listdir
from os.path import join
import tensorflow as tf
import keras.backend as K
from keras.engine import InputSpec, Layer
SYSTEM_EXCEPT_HOOK = sys.excepthook
import matplotlib.pyplot as plt
import shutil
MEGA = 10 ** 6
MEGA_STR = ' ' * MEGA
import os
import glob
import scipy.io
from PIL import Image


class CroppingLike2D(Layer):
    """
    Crop to target.
    If only one `offset` is set, then all dimensions are offset by this amount.
    """
    def __init__(self, target=None, num_classes=None, offset=0, **kwargs):
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = "channels_last"
        self.target_shape = (None, None, None, num_classes)
        self.target = K.zeros(shape=(1,1,1,1))  if target is None else target
        self.input_spec = InputSpec(ndim=4)
        self.offset = offset

    def compute_output_shape(self, input_shape):
        """
        :param input_shape: The images input shape
        :return: The output shape.
        """
        return (input_shape[0],
                self.target_shape[1],
                self.target_shape[2],
                input_shape[3])

    def call(self, inputs, **kwargs):
        """
        :param inputs: The input image.
        :return: The cropped image.
        """
        orig_shape = K.shape(self.target)
        input_image = inputs
        
        return tf.image.crop_to_bounding_box(input_image, self.offset,self.offset, orig_shape[1], orig_shape[2])
                                                                                                                        


def get_listdir_cache():
    folder_dict = {}
    def get_list(folder):
        if folder not in folder_dict:
            folder_dict[folder] = sorted(listdir(folder))
        return folder_dict[folder]
    return get_list

LIST_DIR_CACHE = get_listdir_cache()



def new_except_hook(network):
    def excepthook(*args, **kwargs):
        print("in except hook")
        network.on_exit()
        SYSTEM_EXCEPT_HOOK(*args, **kwargs)
    return excepthook


def get_bilinear_filter(filter_shape, upscale_factor):
    """
    :param filter_shape: The filter shape.
    :param upscale_factor: The upsacle factor for the filter.
    :return: A bilinear filter.
    """
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


def get_model_memory_usage(batch_size, model):
    """
    :return: The memory usage.
    """
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


def display_results(image, label, prediction):
    """
    Plot original image, prediction image and true label.
    :param image: The original image.
    :param label: The true label.
    :param prediction: The networks prediction.
    """

    print("Accuracy : ", np.mean(prediction == label))
    tp, fn, fp = evaluate_iou(label, prediction)
    if tp + fp + fn == 0:
        iou = 1.
    else:
        iou = tp / float(tp + fp + fn)
    print("IOU : ", iou )
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



def make_palette(num_classes):
    """
    -----------Taken from https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_011.pdf-------------
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def convert_sbdd():
    dataset_dir = './benchmark_RELEASE/dataset'
    palette = make_palette(256).reshape(-1)
    for kind in ('cls', 'inst'):
        # collect the inputs
        paths = glob.glob('{}/{}/*.mat'.format(dataset_dir, kind))
        ids = [os.path.basename(p)[:-4] for p in paths]
        for i, idx in enumerate(ids):
            if i % 100 == 0:
                print ("Converting {}th annotation...".format(i))
            # loading the label
            mat = scipy.io.loadmat('{}/{}/{}.mat'.format(dataset_dir, kind, idx))
            label_arr = mat['GT{}'.format(kind)][0]['Segmentation'][0].astype(np.uint8)
            # saving the label
            label_im = Image.fromarray(label_arr)
            label_im.putpalette(palette)
            label_im.save('{}/{}/{}.png'.format(dataset_dir, kind, idx))


# ------------------------------------------ Prepare Data ---------------------------------------------------
if __name__=='__main__':
    convert_sbdd()
    print("finished converting mat files to png, completing the validation dataset from seg11valid.txt")
    complete_seg11valid_set("VOCdevkit/VOC2012/JPEGImages",'benchmark_RELEASE/dataset/img/','benchmark_RELEASE/dataset/seg11valid.txt', '.jpg')
    complete_seg11valid_set("VOCdevkit/VOC2012/SegmentationClass",'benchmark_RELEASE/dataset/cls/','benchmark_RELEASE/dataset/seg11valid.txt', '.png')
