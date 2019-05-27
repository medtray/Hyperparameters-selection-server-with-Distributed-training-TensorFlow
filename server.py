import threading
import socket
import sys

from concurrent.futures import ThreadPoolExecutor
import logging
import time
import itertools
import os
import ast
import json
import os
import pickle
import sys
import distributed_training

counter = itertools.count()
period=3
import shutil
import time
from multiprocessing import Process

import time, threading

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def func(worker,id,params_for_model_string,server_id):
    os.system('python start_training_for_workers.py '+worker+' '+str(id)+' '+params_for_model_string+' '+server_id)

def func2(worker,id,params_for_model_string,server_id,worker_name):
    

    os.system('ssh -i "mykey.pem" ' + worker_name + ' \'cd /home/ec2-user/DistributedTF/code && python start_training_for_workers.py '+worker+' '+str(id)+' '+params_for_model_string+' '+server_id+'\'')

def clear_port(port,worker_name):
    
    os.system('ssh -i "/home/ec2-user/DistributedTF/code/mykey.pem" ' + worker_name + ' \'sudo fuser -k ' + port + '/tcp' + '\'')

def clear_port_local(port):
    
    os.system('sudo fuser -k ' + str(port) + '/tcp')


def handle(connection, address,server_id):


    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("process-%r" % (address,))

    try:
        logger.debug("Connected %r at %r", connection, address)


        while True:
            data = connection.recv(1024)
            if data.decode() == "":
                logger.debug("Socket closed remotely")
                break
            logger.debug("Received data %r", data)

            splitted=data.decode().split("+")

            file_name = "distributed_servers"
            file = open(file_name, "r")
            

            info = []
            for x in file:
                if x[-1] == "\n":
                    info.append(x[:-1])
                else:
                    info.append(x)

            cluster=info[int(server_id)]



            cluster = ast.literal_eval(cluster)
            workers = cluster["worker"]
            master = cluster["master"][0]
            ps = cluster["ps"][0]

            file_name2 = "distributed_servers_names"
            file2 = open(file_name2, "r")
            

            info = []
            for x in file2:
                if x[-1] == "\n":
                    info.append(x[:-1])
                else:
                    info.append(x)

            cluster2 = info[int(server_id)]

            cluster2 = ast.literal_eval(cluster2)
            worker_names = cluster2["worker"]


            if splitted[0]=="DELETE":
                params_for_model_string = splitted[1]
                name_of_model = params_for_model_string
                directory = os.path.dirname(os.getcwd())
                model_dir = directory + "/" + name_of_model
                shutil.rmtree(model_dir)
                
                to_send = "TRUE"
                connection.sendall(to_send.encode())

            if splitted[0]=="TRAIN":
                dist_train=int(splitted[2])

                params_for_model_string=splitted[1]
                params_for_model_dict=ast.literal_eval(params_for_model_string)
                name_of_model=params_for_model_string
                directory = os.path.dirname(os.getcwd())
                model_dir = directory + "/" + name_of_model
                epochs = params_for_model_dict["epochs"]
                DistTrain = distributed_training.distributed_training(params_for_model_dict, file_name, name_of_model,server_id)

                if dist_train:
                    executor = ThreadPoolExecutor(6)
                    params_for_model_string="\""+str(params_for_model_string)+"\""

                    

                    for id,worker in enumerate(workers):
                        #for localhost
                        #executor.submit(func, "worker",id,params_for_model_string,server_id)
                        #for distributed
                        executor.submit(func2, "worker", id,params_for_model_string,server_id,worker_names[id])
                        

                    executor.submit(func, "ps", 0,params_for_model_string,server_id)



                    
                    p = Process(target=func, args=("master", 0, params_for_model_string,server_id,))
                    time.sleep(4)
                    p.start()
                    p.join()
                    

                    DistTrain.evaluate(model_dir)

                    for id,worker in enumerate(workers):
                        ip_port=worker.split(":")
                        port=ip_port[1]
                        #for localhost
                        #executor.submit(clear_port_local, port)
                        # for distributed
                        executor.submit(clear_port, port, worker_names[id])

                    ip_port_ps = ps.split(":")
                    port_ps = ip_port_ps[1]
                    executor.submit(clear_port_local, port_ps)

                else:
                    DistTrain.train_singlethread("master", 0, epochs)

                

                accuracy=DistTrain.test(model_dir)

                



                if accuracy==None:
                    to_send="NONE"
                else:
                    to_send=str(accuracy)
                connection.sendall(to_send.encode())


            logger.debug("Sent data")
    except:
        logger.exception("Problem handling request")
    finally:
        logger.debug("Closing socket")

class Server(object):

    def __init__(self, hostname, port,globalIP,server_id):
        import logging
        self.logger = logging.getLogger("server")
        self.hostname = hostname
        self.port = port
        self.globalIP = globalIP
        self.server_id=server_id




    def start(self):
        import logging
        logging.basicConfig(level=logging.DEBUG)


        executor = ThreadPoolExecutor(5)
        self.logger.debug("listening")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.hostname, self.port))
        self.socket.listen(5)




        while True:
            conn, address = self.socket.accept()
            self.logger.debug("Got connection")

            thread = executor.submit(handle, conn, address, self.server_id)
            #process.daemon = True
            #thread.start()
            self.logger.debug("Started process %r", thread)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    globalIP=sys.argv[1]
    ip = sys.argv[2]
    port = sys.argv[3]
    server_id = sys.argv[4]

    server = Server(ip, int(port),globalIP,server_id)


    try:
        logging.info("Listening")
        server.start()
    except:
        logging.exception("Unexpected exception")
    finally:
        logging.info("Shutting down")
        


