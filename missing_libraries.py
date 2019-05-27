import os
from concurrent.futures import ThreadPoolExecutor
import ast


executor = ThreadPoolExecutor(30)


folder='DistributedTF'
key_path=folder+'/code/mykey.pem'
to_ssh='ssh -i '+'\"'+key_path+'\"'+' '

def func2(name):
    cluster = ast.literal_eval(name)
    workers=cluster["worker"]
    master=cluster["master"]
    for i,worker in enumerate(workers):
        os.system(to_ssh + worker + ' sudo pip install -U scikit-learn')


    os.system(
        to_ssh + master[0] + ' sudo pip install -U scikit-learn')



f = open("distributed_servers_names", "r")
names=[]
for x in f:
    if x[-1]=="\n":
        names.append(x[:-1])
    else:
        names.append(x)

for i in range(len(names)):
    
    func2(names[i])
