from re import X
import tensorflow as tf
import tensorflow.python.keras as keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import math
from matplotlib import pyplot as plt
from copy import copy
import cv2

from sklearn.utils import class_weight
'''
create a world of size 1*1. have two different world states, inside and outside of a circle centered somewhere in the world.

then, we take the trained network, and starting from one section, use gradient descent to let the network move itself inside or outside of the circle as specified by us.

this model learns as it goes!
'''

cv2.namedWindow("bug!")

net = models.Sequential()
net.add(layers.Input((2,)))
net.add(layers.Dense(64,activation='relu'))
net.add(layers.Dense(128,activation='relu'))
net.add(layers.Dense(256,activation='relu'))
net.add(layers.Dense(4,activation='softmax'))

net.compile(loss='binary_crossentropy',metrics=['acc'],optimizer=optimizers.rmsprop_v2.RMSprop(1e-3))

circles = [[0.2,0.2,0.1],[0.2,0.5,0.1],[0.8,0.9,0.1]]


def get_world_from_pos(x,circles):

    res = [0,0,0,0]

    if np.linalg.norm(circles[0][:2] - x[-1]) < circles[0][-1]:
        res = [0,0,0,1]
    elif np.linalg.norm(circles[1][:2] - x[-1]) < circles[1][-1]:
        res = [0,0,1,0]
    elif np.linalg.norm(circles[2][:2] - x[-1]) < circles[2][-1]:
        res = [0,1,0,0]
    else:
        res = [1,0,0,0]

    
    return np.array(res)

def gen_data(samples,circles):
    x = []
    y = []

    center = np.array([0.4,0.4])
    radius = 0.1

    for i in range(samples):
        x.append(np.random.rand(2))
        
        if np.linalg.norm(circles[0][:2] - x[-1]) < circles[0][-1]:
            y.append(np.array([0,0,0,1]))
        elif np.linalg.norm(circles[1][:2] - x[-1]) < circles[1][-1]:
            y.append(np.array([0,0,1,0]))
        elif np.linalg.norm(circles[2][:2] - x[-1]) < circles[2][-1]:
            y.append(np.array([0,1,0,0]))
        else:
            y.append(np.array([1,0,0,0]))

    
    return np.array(x),np.array(y)

data = gen_data(200,circles)

hand_made_data = [[[0.25,0.25],[0.25,0.75],[0.75,0.75],[0.75,0.25]] , [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]]

epochs = 300

net.fit(data[0],data[1],epochs=epochs)
net.compile(loss='binary_crossentropy',metrics=['acc'],optimizer=optimizers.rmsprop_v2.RMSprop(1e-4))
#net.fit(np.array(hand_made_data[0]),np.array(hand_made_data[1]),epochs=epochs)

print(net.predict(np.array([[-10,-10]])))

start_pos = np.array([[0.2,0.5]])#np.ones((1,2))* 0.3

ideal_state = np.array([[0,0,1,0]])

input_1 = layers.Input((1,))

get_pos = layers.Dense(2,name='get_pos',weights=[start_pos],use_bias=False)(input_1)

out = net(get_pos)

get_pos = models.Model(inputs=[input_1],outputs=[out])

#get_pos.layers[1].set_weights(np.ones((1,2)))

get_pos.layers[2].trainable = False

get_pos.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop_v2.RMSprop(1e-3))

get_pos.summary()


def inverse_sigmoid(x):
    return math.log(1 - (1/x))
steps = 150
#print(get_pos.layers[1].weights)
res_x = []
res_y = []


ideal_states = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])

circle = plt.Circle(circles[0][:2],circles[0][-1],color='b')
circle_2 = plt.Circle(circles[1][:2],circles[1][-1],color='r')
circle_3 = plt.Circle(circles[2][:2],circles[2][-1],color='g')

flag = False


gathered_x = list(data[0])
gathered_y = list(data[1])

state_thresh = 0.001

num_states = 2000

for k in range(num_states):

    j = ideal_states[k % len(ideal_states)]
    # if np.random.rand(1) < 0.3 and not flag:
    #     get_pos.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop_v2.RMSprop(1e-2))
    #     flag = True
    # elif flag:
    #     get_pos.compile(loss='binary_crossentropy',optimizer=optimizers.rmsprop_v2.RMSprop(1e-3))
    #     flag = False
    print("MOVING TO STATE: ",j)
    for i in range(steps):

            
        get_pos.fit(np.array([[1]]),np.array([j]),epochs=5,verbose=0)
        #print(get_pos.layers[1].weights,get_pos(np.array([[1]])))
        pos = get_pos.layers[1].weights


        x = min(max(pos[0][0][0],0),1)
        y = min(max(pos[0][0][1],0),1)
        get_pos.layers[1].set_weights([np.array([[x,y]])])

        gathered_x.append(np.array([pos[0][0][0],pos[0][0][1]]))
        gathered_y.append(get_world_from_pos(np.array([x,y]),circles))

        res_x.append(x)
        res_y.append(y)

        #print(j)
        fig,ax = plt.subplots()
        plt.xlim([0,1])
        plt.ylim([0,1])

        ax.add_patch(copy(circle))
        ax.add_patch(copy(circle_2))
        ax.add_patch(copy(circle_3))

        ax.plot(res_x,res_y)
        ax.plot([res_x[-1]],[res_y[-1]],'yo')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imshow("bug!",img)
        cv2.waitKey(1)

        plt.close(fig)

        #go to next state if the network thinks its within a threshold of the goal state
        if(np.linalg.norm(net(np.array([[x,y]]))[0] - np.array([j])) < state_thresh):
            break


    net.trainable = True

    net.fit(np.array(gathered_x),np.array(gathered_y),epochs=10)

    net.trainable = False

    










