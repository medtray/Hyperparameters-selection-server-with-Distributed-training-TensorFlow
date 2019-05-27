import ast
import json
import os
import pickle
import sys

import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.propagate = False
tf.logging.set_verbosity(tf.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import model_definition

class distributed_training(object):

    def __init__(self,params,cluster_info,model_dir_name,server_id):
        self.params=params
        self.cluster_info=cluster_info
        self.model_dir_name=model_dir_name
        self.server_id=server_id

    def _get_session_config_from_env_var(self):
        """Returns a tf.ConfigProto instance that has appropriate device_filters
        set."""

        tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

        if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
                    'index' in tf_config['task']):
            # Master should only communicate with itself and ps
            if tf_config['task']['type'] == 'master':
                return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
            # Worker should only communicate with itself and ps
            elif tf_config['task']['type'] == 'worker':
                return tf.ConfigProto(device_filters=[
                    '/job:ps',
                    '/job:worker/task:%d' % tf_config['task']['index']
                ])
        return None

    def train(self,server_role, server_number, epochs):

        directory = os.path.dirname(os.getcwd())

        if server_role == "master" or server_role == "ps":
            file_pi = open(directory + '/chunks/chunk_master.obj', 'r')
            train_data = pickle.load(file_pi)
            file2 = open(directory + '/chunks/chunk_master_labels.obj', 'r')
            train_labels = pickle.load(file2)

        else:

            file_pi = open(directory + '/chunks/chunk' + str(server_number) + '.obj', 'r')
            train_data = pickle.load(file_pi)
            file2 = open(directory + '/chunks/chunk_labels' + str(server_number) + '.obj', 'r')
            train_labels = pickle.load(file2)

        # hyper parameters
        batch_size = 100


        file = open(self.cluster_info, "r")
        info = []
        for x in file:
            if x[-1] == "\n":
                info.append(x[:-1])
            else:
                info.append(x)

        cluster = info[int(self.server_id)]
        #cluster = file.read()
        cluster = ast.literal_eval(cluster)

        os.environ['TF_CONFIG'] = json.dumps(
            {'cluster': cluster,
             'task': {'type': server_role, 'index': server_number}})

        directory = os.path.dirname(os.getcwd())

        
        model_dir = directory + '/' + self.model_dir_name
        to_print="start from "+server_role+" "+str(server_number)
        print(to_print)
        # create run config for estimator
        run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=2000, keep_checkpoint_max=4,
                                            session_config=self._get_session_config_from_env_var())
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)

        # create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_definition.cnn_model_fn2,
            model_dir=model_dir,
            config=run_config,
            params={
                'param': self.params,

            }

        )


        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=5,
            num_epochs=epochs,
            shuffle=True)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            num_epochs=1,
            shuffle=False)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
        to_print = "finish from " + server_role + " " + str(server_number)
        print(to_print)

        return



    def train_singlethread(self,server_role, server_number, epochs):

        directory = os.path.dirname(os.getcwd())

        if server_role == "master" or server_role == "ps":
            file_pi = open(directory + '/chunks/chunk_master.obj', 'r')
            train_data = pickle.load(file_pi)
            file2 = open(directory + '/chunks/chunk_master_labels.obj', 'r')
            train_labels = pickle.load(file2)

        else:

            file_pi = open(directory + '/chunks/chunk' + str(server_number) + '.obj', 'r')
            train_data = pickle.load(file_pi)
            file2 = open(directory + '/chunks/chunk_labels' + str(server_number) + '.obj', 'r')
            train_labels = pickle.load(file2)



        directory = os.path.dirname(os.getcwd())

        model_dir = directory + '/'+self.model_dir_name
        # create run config for estimator
        run_config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=2000, keep_checkpoint_max=4)
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=1000)

        # create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_definition.cnn_model_fn2,
            model_dir=model_dir,
            config=run_config,
            params={
                'param': self.params,

            }

        )


        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=5,
            num_epochs=epochs,
            shuffle=True)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            num_epochs=1,
            shuffle=False)
        # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
        # tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

        # train model
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=train_labels.shape[0] / 5 * epochs,
            max_steps=None,
            hooks=[logging_hook]
        )

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        return


    def test(self,model_dir):

        directory = os.path.dirname(os.getcwd())

        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_definition.cnn_model_fn2,
            model_dir=model_dir,
            params={
                'param': self.params,

            }

        )

        file_pi = open(directory + '/chunks/chunk_validation.obj', 'r')
        train_data = pickle.load(file_pi)
        file2 = open(directory + '/chunks/chunk_validation_labels.obj', 'r')
        train_labels = pickle.load(file2)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            num_epochs=1,
            shuffle=False)
        predict_results = list(mnist_classifier.predict(input_fn=predict_input_fn))
        predicted_classes = [p['class_id'] for p in predict_results]
        accuracy_of_testing = accuracy_score(train_labels, predicted_classes)

        return accuracy_of_testing


    def evaluate(self,model_dir):

        directory = os.path.dirname(os.getcwd())

        mnist_classifier = tf.estimator.Estimator(
            model_fn=model_definition.cnn_model_fn2,
            model_dir=model_dir,
            params={
                'param': self.params,

            }

        )

        file_pi = open(directory + '/chunks/chunk_master.obj', 'r')
        train_data = pickle.load(file_pi)
        file2 = open(directory + '/chunks/chunk_master_labels.obj', 'r')
        train_labels = pickle.load(file2)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            num_epochs=1,
            shuffle=False)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)



        return
