from keras.objectives import *
from keras.metrics import binary_crossentropy
import keras.backend as K
import tensorflow as tf


# Softmax cross-entropy loss function for pascal voc segmentation
# and models which do not perform softmax.
# tensorlow only
def softmax_sparse_crossentropy_ignoring_last_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)
    
    #v = tf.Print(cross_entropy_mean, [cross_entropy_mean])
    #v.eval(session=tf.Session())
    return cross_entropy_mean


def sparse_cross_entropy(y_true, y_pred):
    #comparison = tf.equal(y_true, 255)
    #keeps  = tf.where(comparison)
    #import pdb
    #pdb.set_trace()
    #y_t = K.gather(y_true, keeps)
    #y_p = K.gather(y_pred, keeps)
    class_weights = K.constant([1]*21+[0])
    weights = K.gather(class_weights, y_true)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.squeeze(y_true, squeeze_dims=[3]),
                                                                     logits=y_pred,weights=weights)
    loss_mean = -tf.reduce_mean(loss)
    return loss_mean

# Softmax cross-entropy loss function for coco segmentation
# and models which expect but do not apply sigmoid on each entry
# tensorlow only
def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth,
                                        predictions,
                                        from_logits=True),
                  axis=-1)
