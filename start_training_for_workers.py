import ast
import json
import os
import pickle
import sys

import tensorflow as tf
from sklearn.metrics import accuracy_score

tf.logging.set_verbosity(tf.logging.INFO)

import model_definition
import distributed_training

server_role = sys.argv[1]
server_number = sys.argv[2]
params_for_model_string = sys.argv[3]
server_id = sys.argv[4]
print(params_for_model_string)


if server_role=="worker":
    params_for_model_string = params_for_model_string[1:len(params_for_model_string) - 1]
    inter=params_for_model_string.split(',')
    n='{'
    for i in range(len(inter)):
        inter2=inter[i].split(':')
        if i==0:
            key='\''+inter2[0]+'\''

        else:
            key = '\'' + inter2[0][1:] + '\''
        n=n+key+':'+inter2[1]+', '

    n=n[0:len(n)-2]

    n=n+'}'

    params_for_model_string=n

    print(params_for_model_string)

params_for_model_dict=ast.literal_eval(str(params_for_model_string))



epochs=params_for_model_dict["epochs"]

cluster_info="distributed_servers"
DistTrain=distributed_training.distributed_training(params_for_model_dict,cluster_info,params_for_model_string,server_id)
DistTrain.train(server_role,int(server_number),epochs)