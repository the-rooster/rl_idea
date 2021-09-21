from re import X
import tensorflow as tf
import tensorflow.python.keras as keras
from keras import models
from keras import layers
import numpy as np
import math
from matplotlib import pyplot as plt
from copy import copy

'''
create a world of size 1*1. have two different world states, inside and outside of a circle centered somewhere in the world.

then, we take the trained network, and starting from one section, use gradient descent to let the network move itself inside or outside of the circle as specified by us.
'''

net = models.Sequential()
net.add(layers.Input((2,)))
net.add(layers.Dense(32,activation='relu'))
net.add(layers.Dense(64,activation='relu'))
net.add(layers.Dense(128,activation='relu'))
net.add(layers.Dense(4,activation='softmax'))

net.compile(loss='binary_crossentropy',metrics=['acc'])

def gen_data(samples):
    x = []
    y = []

    center = np.array([0.4,0.4])
    radius = 0.1

    for i in range(samples):
        x.append(np.random.rand(2))
        
        if np.linalg.norm(center - x[-1]) < radius:
            y.append(np.array([0,0,0,1]))
        else:
            y.append(np.array([1,0,0,0]))

    
    return np.array(x),np.array(y)

data = gen_data(1000)

hand_made_data = [[[0.25,0.25],[0.25,0.75],[0.75,0.75],[0.75,0.25]] , [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]]

epochs = 100

net.fit(data[0],data[1],epochs=epochs)
#net.fit(np.array(hand_made_data[0]),np.array(hand_made_data[1]),epochs=epochs)

print(net.predict(np.array([[-10,-10]])))

start_pos = np.ones((1,2))* 0.8

ideal_state = np.array([[0,0,1,0]])

input_1 = layers.Input((1,))

get_pos = layers.Dense(2,name='get_pos',weights=[start_pos],use_bias=False)(input_1)

out = net(get_pos)

get_pos = models.Model(inputs=[input_1],outputs=[out])

#get_pos.layers[1].set_weights(np.ones((1,2)))

get_pos.layers[2].trainable = False

get_pos.compile(loss='binary_crossentropy')

get_pos.summary()


def inverse_sigmoid(x):
    return math.log(1 - (1/x))
steps = 150
#print(get_pos.layers[1].weights)
res_x = []
res_y = []


ideal_states = np.array([[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1],[1,0,0,0],[0,0,0,1]])

circle = plt.Circle((0.4,0.4),0.1,color='b')

flag = True


for j in ideal_states:
    for i in range(steps):
        get_pos.fit(np.array([[1]]),np.array([j]),epochs=5,verbose=0)
        print(get_pos.layers[1].weights,get_pos(np.array([[1]])))
        pos = get_pos.layers[1].weights


        x = min(max(pos[0][0][0],0),1)
        y = min(max(pos[0][0][1],0),1)
        get_pos.layers[1].set_weights([np.array([[x,y]])])

        res_x.append(x)
        res_y.append(y)

    print(j)
    fig,ax = plt.subplots()
    plt.xlim([0,1])
    plt.ylim([0,1])

    ax.add_patch(copy(circle))

    ax.plot(res_x,res_y)
    ax.plot([res_x[-1]],[res_y[-1]],'ro')

    plt.show()










