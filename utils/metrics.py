import keras.backend as K
import tensorflow as tf


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    """
    :param y_true: The true label.
    :param y_pred: The predicted label.
    :return: The accuracy ignoring last label.
    """
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))
    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


def Mean_IOU(y_true, y_pred):
    """
    :param y_true: The true label.
    :param y_pred: The predicted label.
    :return: Mean IOU.
    """
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.squeeze(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(true_pixels, 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        if K.eval(K.equal(K.sum(union, axis=1), 0)):
            iou.append(K.sum(union))
            continue
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(tf.cast(iou, tf.float32))
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)
