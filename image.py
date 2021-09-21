from re import X
import tensorflow as tf
import tensorflow.python.keras as keras
import tensorflow.python.keras.backend as K
from keras import callbacks
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import math
from matplotlib import pyplot as plt
import cv2

'''
use images as the world state, with position as the agent state.

NOT WORKING! network is bad :(
    should look into how decoder networks be working (with sparse binary maps)
    either that or just make the target bigger
'''
def sparse_crossentropy_masked(y_true, y_pred):
    # y_pred = K.flatten(y_pred)
    # y_true = K.flatten(y_true)
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    return K.mean(K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked))

net = models.Sequential()
net.add(layers.Input((2,)))
net.add(layers.Dense(4))
net.add(layers.LeakyReLU(0.2))
# net.add(layers.Dense(32))
# net.add(layers.LeakyReLU(0.2))
# net.add(layers.Dense(64))
net.add(layers.LeakyReLU(0.2))
net.add(layers.Dense(16))
net.add(layers.LeakyReLU(0.2))
net.add(layers.Reshape((4,4,1)))
net.add(layers.Conv2D(32,3,padding='same',activation='relu'))
net.add(layers.LeakyReLU(0.2))
net.add(layers.UpSampling2D(2))
net.add(layers.Conv2D(64,3,padding='same',activation='relu'))
net.add(layers.LeakyReLU(0.2))
net.add(layers.UpSampling2D(2))
#net.add(layers.BatchNormalization())
net.add(layers.Conv2D(128,3,padding='same',activation='relu'))
net.add(layers.LeakyReLU(0.2))
net.add(layers.UpSampling2D(2))
#net.add(layers.UpSampling2D(2))
#net.add(layers.BatchNormalization())
#net.add(layers.UpSampling2D(2))
#net.add(layers.BatchNormalization())
net.add(layers.Conv2D(1,3,padding='same',activation='sigmoid'))
#net.add(layers.Reshape((32,32)))

net.compile(loss='binary_crossentropy',metrics=['acc'],optimizer=optimizers.rmsprop_v2.RMSProp(1e-3))

cv2.namedWindow("dataGen")
def gen_data(samples):
    x = []
    y = []

    for i in range(samples):
        x_i = np.random.rand(2)

        x.append(x_i)
        
        y_i = np.zeros((32,32,1))*(1/2)

        x_i *= 31
        x_i = np.int32(x_i)

        y_i[x_i[0] - 2:x_i[0] + 2,x_i[1] - 2:x_i[1] + 2] = 1

        cv2.imshow("dataGen",cv2.resize(y_i,(500,500)))
        cv2.waitKey(1)
        #y_i[x_i[0],x_i[1]] = 1
        y.append(y_i)

    
    return np.array(x),np.array(y)

data = gen_data(500)

epochs = 100

#net.fit(data[0],data[1],epochs=epochs)
net.fit(np.array(data[0]),np.array(data[1]),epochs=epochs,callbacks=[callbacks.ReduceLROnPlateau(monitor='loss')])

print(net.predict(np.array([[-10,-10]])))

start_pos = np.ones((1,2))* 0.2

ideal_state = np.array([[0,0,1,0]])

input_1 = layers.Input((1,))

get_pos = layers.Dense(2,name='get_pos',weights=[start_pos],use_bias=False)(input_1)

out = net(get_pos)

get_pos = models.Model(inputs=[input_1],outputs=[out])

#get_pos.layers[1].set_weights(np.ones((1,2)))

out.trainable = False

get_pos.compile(loss='mse',optimizer=optimizers.rmsprop_v2.RMSProp(1e-2))

get_pos.summary()


def get_world(x):
    y_i = np.zeros((32,32,1))*(1/2)

    x *= 31
    x_i = np.int32(x)

    y_i[x_i[0] - 2:x_i[0] + 2,x_i[1] - 2:x_i[1] + 2] = 1

    return y_i

steps = 150
#print(get_pos.layers[1].weights)
res_x = []
res_y = []


ideal_states = np.array([get_world(np.array([0.8,0.8])),get_world(np.array([0.1,0.9])),get_world(np.array([0.8,0.2])),get_world(np.array([0.5,0.1]))])


cv2.imshow("test1",cv2.resize(np.float32(net(np.array([[0.1,0.1]]))[0]),(500,500)))
cv2.imshow("test2",cv2.resize(np.float32(net(np.array([[0.5,0.9]]))[0]),(500,500)))
cv2.waitKey()

cv2.namedWindow("Goal")
cv2.namedWindow("Internal")
cv2.namedWindow("Current")

num_times = 1000

for k in range(num_times):

    j = ideal_states[k % len(ideal_states)]
    for i in range(steps):
        get_pos.fit(np.array([[1]]),np.array([j]),epochs=5,verbose=0)
        print(get_pos.layers[1].weights)
        pos = get_pos.layers[1].weights


        x = min(max(pos[0][0][0],0),1)
        y = min(max(pos[0][0][1],0),1)
        get_pos.layers[1].set_weights([np.array([[x,y]])])

        res_x.append(x)
        res_y.append(y)

        x_i = np.array([x,y]) * 31
        x_i = np.int32(x_i)

        img = np.zeros((32,32,1))
        img[x_i[0],x_i[1]] = 1

        internal_world = np.float32(net(np.array([[x,y]]))[0])

        
        print(internal_world.shape)

        cv2.imshow("Goal",cv2.resize(j,(400,400)))
        cv2.imshow("Internal",cv2.resize(internal_world,(400,400)))
        cv2.imshow("Current",cv2.resize(img,(400,400)))
        cv2.waitKey(1)









