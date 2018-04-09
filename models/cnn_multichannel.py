from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D,Activation
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import tensorflow as tf
import numpy as np

x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')
weights=np.load('embedding_weights.npy')

#input parameters
max_length=x_train.shape[1]
#embedding layer parameters
input_dim_embedding=18779
out_dim_embedding=300
#conv parameters
s1=3
s2=4
s3=5
stride=1
num_filter=1
#output_dim
class_number=1
#model parameters
batch_size=50
num_epochs=10

model_input=Input(shape=(max_length,))
z1=Embedding(input_dim=input_dim_embedding,output_dim=out_dim_embedding,input_length=max_length,name="embedding1",trainable=True)(model_input)
z2=Embedding(input_dim=input_dim_embedding,output_dim=out_dim_embedding,input_length=max_length,name="embedding2",trainable=False)(model_input)
z=Concatenate(axis=-1)([z1,z2])
conv1 = Convolution1D(filters=num_filter,
                     kernel_size=s1,
                     padding="valid",
                     activation="relu",
                     strides=1)(z)
pool1 = MaxPooling1D(pool_size=int(conv1.shape[1]),strides=stride,padding='valid')(conv1)
pool1 = Flatten()(pool1)

conv2 = Convolution1D(filters=num_filter,
                     kernel_size=s2,
                     padding="valid",
                     activation="relu",
                     strides=1)(z)
pool2 = MaxPooling1D(pool_size=int(conv2.shape[1]),strides=stride,padding='valid')(conv2)
pool2 = Flatten()(pool2)

conv3 = Convolution1D(filters=num_filter,
                     kernel_size=s3,
                     padding="valid",
                     activation="relu",
                     strides=1)(z)
pool3 = MaxPooling1D(pool_size=int(conv3.shape[1]),strides=stride,padding='valid')(conv3)
pool3 = Flatten()(pool3)

fc_1=Concatenate()([pool1,pool2,pool3])
#fc_1=Dropout(0.5)(fc_1)
out=Dense(class_number)(fc_1)
model_output=Activation('sigmoid')(out)

model=Model(inputs=model_input,outputs=model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
embedding_layer1=model.get_layer("embedding1")
embedding_layer1.set_weights([weights])
embedding_layer2=model.get_layer("embedding2")
embedding_layer2.set_weights([weights])
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,validation_data=(x_test, y_test))
model.summary()