import os
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging

#logging.basicConfig(level=logging.INFO)
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.logging._logger.propagate = False

#logging.propagate = False
def cnn_model_fn2(features, labels, mode, params):
    
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    input_size = 100
    n_output_classes = 2

    param=params['param']
    nb_filter1=param['nb_filter1']
    nb_filter2 = param['nb_filter2']
    nb_filter3 = param['nb_filter3']

    inputs = tf.reshape(features['x'], shape=[-1, input_size, input_size, 3])

    conv1 = tf.layers.conv2d(inputs, filters=nb_filter1, kernel_size=5, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    
    conv2 = tf.layers.conv2d(pool1, filters=nb_filter2, kernel_size=5, padding='same', activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    
    flat3 = tf.layers.flatten(pool2)

    
    dense4 = tf.layers.dense(flat3, units=nb_filter3, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(dense4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    
    logits = tf.layers.dense(dropout4, units=n_output_classes)

    
    predicted_classes = tf.argmax(logits, axis=1)
    predictions = {
        'class_id': tf.cast(predicted_classes, dtype=tf.int32),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["class_id"])}


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=0.00000001)
        # optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=2)
        # grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())


        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

