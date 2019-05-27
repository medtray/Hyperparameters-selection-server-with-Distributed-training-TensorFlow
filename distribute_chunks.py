import os
from concurrent.futures import ThreadPoolExecutor
import ast


executor = ThreadPoolExecutor(30)


folder='DistributedTF'
key_path=folder+'/code/mykey.pem'
to_ssh='ssh -i '+'\"'+key_path+'\"'+' '
to_scp_chunk='scp -i '+key_path+' -r '+folder+'/chunks/chunk'
to_scp_chunk_labels='scp -i '+key_path+' -r '+folder+'/chunks/chunk_labels'
to_scp_code='scp -i '+key_path+' -r '+folder+'/code '

to_scp_chunk_master='scp -i '+key_path+' -r '+folder+'/chunks/chunk_master.obj '
to_scp_chunk_master_labels='scp -i '+key_path+' -r '+folder+'/chunks/chunk_master_labels.obj '

to_scp_chunk_validation='scp -i '+key_path+' -r '+folder+'/chunks/chunk_validation.obj '
to_scp_chunk_validation_labels='scp -i '+key_path+' -r '+folder+'/chunks/chunk_validation_labels.obj '

def func2(name,send_chunks):
    cluster = ast.literal_eval(name)
    workers=cluster["worker"]
    master=cluster["master"]
    for i,worker in enumerate(workers):
        if send_chunks:
            os.system(to_ssh + worker + ' sudo rm -r /home/ec2-user/DistributedTF')
            os.system(to_ssh + worker + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF')
            os.system(to_ssh + worker + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF/chunks')
            os.system(
                to_scp_chunk+str(i)+".obj " + worker + ":/home/ec2-user/DistributedTF/chunks/chunk"+str(i)+".obj")
            os.system(
                to_scp_chunk_labels + str(i) + ".obj " + worker + ":/home/ec2-user/DistributedTF/chunks")

        os.system(to_ssh + worker + ' sudo rm -r /home/ec2-user/DistributedTF/code')
        os.system(to_scp_code + worker + ":/home/ec2-user/DistributedTF/code")

    if send_chunks:
        os.system(to_ssh + master[0] + ' sudo rm -r /home/ec2-user/DistributedTF')
        os.system(
            to_ssh + master[0] + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF')
        os.system(
            to_ssh + master[0] + ' sudo mkdir -m 777 /home/ec2-user/DistributedTF/chunks')
        os.system(
            to_scp_chunk_master + master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_master.obj")
        os.system(
            to_scp_chunk_master_labels + master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_master_labels.obj")

        os.system(
            to_scp_chunk_validation +
            master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_validation.obj")
        os.system(
            to_scp_chunk_validation_labels +
            master[0] + ":/home/ec2-user/DistributedTF/chunks/chunk_validation_labels.obj")
    os.system(to_ssh + master[0] + ' sudo rm -r /home/ec2-user/DistributedTF/code')
    os.system(
        to_scp_code + master[0] + ":/home/ec2-user/DistributedTF/code")


f = open("distributed_servers_names", "r")
names=[]
for x in f:
    if x[-1]=="\n":
        names.append(x[:-1])
    else:
        names.append(x)

send_chunks=True
for i in range(len(names)):
  
    func2(names[i],send_chunks)




