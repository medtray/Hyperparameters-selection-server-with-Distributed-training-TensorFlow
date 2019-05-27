from datetime import time

import numpy as np
import socket
from multiprocessing import Pool
import multiprocessing.pool
from collections import Counter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from random import randint
import random
import threading
import time
import itertools 
import threading


class Client:
    #parameters inititiation
    def __init__(self,server_avail,server_list):
        self.global_variable = 0
        self.server_list = server_list
        self.server_avail = server_avail
        self.acc = [0,0,0,0] #zero at the begining for 50,100,150,200 epoches
        self.best_model = []
        self.writing_lock = threading.Lock()

    #handle function
    def handle(self,host, port,params,server_id,server_list):
        EPOCHS = 5
        dist_train="1"
        
        model_acc = []
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            for i in range(4):
                #make the dictionary and message to send for node with id = server_node
                d={'nb_filter1':params[0],'nb_filter2':params[1],'nb_filter3':params[2],'epochs':EPOCHS}
                d_string=str(d)
                to_send="TRAIN+"+d_string+'+'+dist_train+'+'+str(server_id)
                #send the message to the node
                s.send(to_send.encode('ascii'))
                print(to_send, '...')
                #wait to recieve a response
                data = s.recv(1024)
                response = float(data.decode('ascii'))
                #if the acc is bigger then continue and if it is last epoch then store the whole model
                if(response> self.acc[i]):
                    #save current accuracy for later storage
                    model_acc.append(response)
                    if(i == 3): #if it is the last epoch/message --> save the best model then!
                        current_best = self.best_model

                        

                        print('passed lock check')
                        #lock to write
                        self.writing_lock.acquire()
                        self.acc = model_acc
                        self.best_model = [server_id,params[0],params[1],params[2]]
                        #release the lock
                        self.writing_lock.release()
                        print('release lock ')
                        
                        #remove previous best model data folder --> send a message for that
                        s.close() #close current connection
                        print('socket closed')
                        #open a new connection
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        
                        print((server_list[current_best[0]][0],int( server_list[current_best[0]][1])))
                        s.connect((server_list[current_best[0]][0],int( server_list[current_best[0]][1])))
                        print('connected')
                        d={'nb_filter1':current_best[1],'nb_filter2':current_best[2],'nb_filter3':current_best[3],'epochs':EPOCHS}
                        d_string=str(d)
                        to_send="DELETE+"+d_string+'+'+dist_train+'+'+str(current_best[0])
                        print('message is ready!')
                        s.send(to_send.encode('ascii'))
                        print(to_send, '...')
                        data = s.recv(1024)
                        print(data, '...')
                        s.close()
                else: #send delete to the node and go on!
                    to_send="DELETE+"+d_string+'+'+dist_train+'+'+str(server_id)
                    #send the message to the node
                    s.send(to_send.encode('ascii'))
                    print(to_send, '...')
                    data = s.recv(1024)
                    print(data, '...')
                    #exit the while loop to end this thread
                    break
                    

        except:
            #close the connection and make the current server available
            s.close() 
            self.server_avail[server_id] = 1
            #if the lock has not been released, just release it
            if(self.writing_lock.locked()):
                self.writing_lock.release()

        s.close()
        #At the end set this server available
        self.server_avail[server_id] = 1




#making combination of filters
nb = [[2],[2,3],[5,6]]
filters = list(itertools.product(*nb))

#reading list of servers from a file
file = open('server_list', 'r')  #read the address file
server_list = []
server_avail = [] #a vector to see if the server is available or not
#add nodes address to nodes_list without a key for now
for lines in file:
    result = lines.find('\n') #remove \n if ther is any
    if(result>-1):
        lines = lines[0:result]
    res = lines.split(",")
    server_list.append([res[0],res[1]])
    server_avail.append(1)




#define number of active threads
executor = ThreadPoolExecutor(5)
#make an object of Client Class
cl = Client(server_avail,server_list)


#this is the main body of client...we continue checking available server and at the same time recieving data from servers
#1-when we find an available server we send parameters set and ask to do the training --> set the server to UNAVAILABLE
#2-if the returns acc is less than what we stored before, then we send delete to server and wait for confirmation --> set the node to AVAILABLE
#3-if the server returns acc greater than any node before, we send next call for epoch(50,100,150,200) 
a = time.time()
while(len(filters)>0):
    #if there is at least one server available, sumbit a thread to it
    if(sum(cl.server_avail)>0):
        print(cl.best_model)
        print(cl.acc)
        #set server_avail to zero
        index = cl.server_avail.index(1); cl.server_avail[index] = 0
        #get the topest set of params and submit a thread to run it
        if len(filters)>0:
            params = filters.pop()
            thread = executor.submit(cl.handle, server_list[index][0], int(server_list[index][1]),params,index,server_list)
    
#wait till all threads finish their jobs
while(sum(cl.server_avail)< len(cl.server_avail)):
    
    continue


print('finished  ',cl.best_model)
print('time consumed in Sec : ',time.time() - a)
