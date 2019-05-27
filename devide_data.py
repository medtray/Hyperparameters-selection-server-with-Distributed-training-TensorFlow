import pickle
import numpy as np
import tensorflow as tf
import os

directory = os.path.dirname(os.getcwd())

file_pi = open(directory+'/chunks/AllStopSignData.obj', 'r')
AllRedData = pickle.load(file_pi)
file_pi = open(directory+'/chunks/AllNonStopSignData.obj', 'r')
AllNonRedData = pickle.load(file_pi)

train_labels = np.concatenate(
        (np.ones(len(AllRedData), dtype=np.int32), np.zeros(len(AllNonRedData), dtype=np.int32)), axis=0)

train_data = tf.concat((AllRedData, AllNonRedData), axis=0)
train_data = tf.Session().run(train_data).astype(dtype=np.float32)
train_data = train_data / 255

indices=np.random.permutation(len(train_data))
validation_set_end=int(0.2*len(indices))

slice=indices[0:validation_set_end]

validation_data=train_data[slice]
validation_labels=train_labels[slice]

slice=indices[validation_set_end::]

train_data=train_data[slice]
train_labels=train_labels[slice]


file=open(directory+'/chunks/chunk_validation.obj','w')
file_labels=open(directory+'/chunks/chunk_validation_labels.obj','w')
pickle.dump(validation_data,file)
pickle.dump(validation_labels, file_labels)

file=open(directory+'/chunks/chunk_master.obj','w')
file_labels=open(directory+'/chunks/chunk_master_labels.obj','w')
pickle.dump(train_data,file)
pickle.dump(train_labels, file_labels)

nb_workers=2
total=len(train_data)

indices=np.random.permutation(len(train_data))

for i in range(nb_workers):

    start=int(float(i)/nb_workers*total)
    end=int(float(i+1)/nb_workers*total)
    slice=indices[start:end]

    chunk=train_data[slice]
    labels=train_labels[slice]
    file=open(directory+'/chunks/chunk'+str(i)+".obj",'w')
    file_labels=open(directory+'/chunks/chunk_labels'+str(i)+".obj",'w')
    pickle.dump(chunk,file)
    pickle.dump(labels, file_labels)
