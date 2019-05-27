
import os
from concurrent.futures import ThreadPoolExecutor
import ast

executor = ThreadPoolExecutor(30)
def func(global_ip,local_ip,port,name,server_id):

    os.system('ssh -i "mykey.pem" '+name+ ' \'cd /home/ec2-user/DistributedTF/code && python < server.py - '+global_ip+' '+local_ip+' '+str(port)+' '+str(server_id)+'\'')


f = open("list_of_servers", "r")
info=[]
for x in f:
    if x[-1]=="\n":
        info.append(x[:-1])
    else:
        info.append(x)


file_name2 = "distributed_servers_names"
file2 = open(file_name2, "r")

names = []
for x in file2:
    if x[-1] == "\n":
        names.append(x[:-1])
    else:
        names.append(x)




for i in range(len(info)):
    to_contact = info[i].split(":")
    cluster2 = names[i]

    cluster2 = ast.literal_eval(cluster2)
    server_name = cluster2["master"][0]
    executor.submit(func, to_contact[0],to_contact[1],to_contact[2],server_name,i)



