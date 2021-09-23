from numpy.random import seed
seed(242)
import tensorflow as tf
#import gym
import os
import random

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
import numpy as np
import scipy
import uuid
import shutil

#import pandas as pd
#import matplotlib.pyplot as plt


import tensorflow.keras.backend as K
#import gym_game
import math

from ddpg import OUActionNoise
from ddpg import Buffer


input_shape = (1,) 
outputs = 2

upper_bound=27
lower_bound=-27


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    #last_init = tf.random_uniform_initializer(minval=0.01, maxval=0.99)

    inputs = layers.Input(shape=(3,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh")(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(3))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(2))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

def create_model(learning_rate, regularization_factor):
  model = Sequential([
    Dense(64, input_shape=input_shape, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(outputs, activation='linear', kernel_regularizer=l2(regularization_factor))
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=masked_huber_loss(0.0, 1.0))
  return model

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.02
actor_lr = 0.01

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(10000, 64)



from pyrep import PyRep
import numpy as np
import time
import random

def getSimulatorState(pyrep):
	try:
	    _, s, _, _ = pyrep.script_call(function_name_at_script_name="getState@control_script",
	                                    script_handle_or_type=1,
	                                    ints=(), floats=(), strings=(), bytes="")
	except:
	    print("Couldn't get state from VRep")
	    s = [0.0] * 7
	state = {}
	state["ball_pos"] = s[0:3]
	state["ball_vel"] = s[3:6]
	state["goal"]     = s[6]
	return state
 
def setupEnvironment(pyrep):
	offset =0 #np.random.uniform(-0.1, 0.1, size=None)
	_, _, _, _ = pyrep.script_call(function_name_at_script_name="setupEnvironment@control_script",
	                                script_handle_or_type=1,
	                                ints=(), floats=[offset], strings=(), bytes="")
	pyrep.step()
	return getSimulatorState(pyrep)

def kickBall(pyrep, vx, vy):
	_, _, _, _ = pyrep.script_call(function_name_at_script_name="applyKick@control_script",
	                                script_handle_or_type=1,
	                                ints=(), floats=[vx, vy], strings=(), bytes="")
'''
def moveGoalKeeper(pyrep,predict):
	if predict == "go_left":
		vx = random.choice([65,40,84])
	else:
		vx = random.choice([-84,-65,-40])
	vy = 0
	_, _, _, _ = pyrep.script_call(function_name_at_script_name="goalKeeper@control_script",
	                                script_handle_or_type=1,
	                                ints=(), floats=[vx, vy], strings=(), bytes="")
'''
def stepEnvironment(pyrep):
	state = getSimulatorState(pyrep)
	# Save the state or somehow use it for training

	pyrep.step()
	return state


def haveModelPredictValues(step):
	# Do your RL thing here
	#vel_x, vel_y = (np.random.uniform(-50.0, 50.0, size=None), np.random.uniform(-100.0, -50.0, size=None))
	#vel_x=0
	if (step<5):
		#explore
		vel_x=random.choice(Rx)
	else:
		#exploit
		vel_x=Rx[Q.index(max(Q))]

	vel_y=-75.0
	print("step: ",step)
	#print(vel_x)
	return vel_x, vel_y

def epochFinished(state):
	if state["goal"] == 1:
	    return True
	return False

def get_value(rew,q):
	#print("----- ")
	val=(1-alpha)*q+alpha*(rew)	
	return val

pyrep = PyRep()
pyrep.launch("/home/maitry/Downloads/vrep_scene_1.ttt", headless=False)
alpha=0.3
gamma=0.9
Q=[0,0,0,0,0,0]
Rx=[-30,30,20,-20,-15,15]
iterations = 300
training_data = []
for step in range(iterations):
	c_data = []
	pyrep.start()
	max_steps = 300
	for i in range(max_steps):
		#print(step)
		if i == 0:
			state = setupEnvironment(pyrep)
			tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state["ball_pos"]), 0)
			print(state["ball_pos"])
			prev=state["ball_pos"]

			c_data = c_data + state["ball_pos"]
		if i == 1:
			action = policy(tf_prev_state, ou_noise)
			w1=action[0][0]
			w2=action[0][1]		
			vx, vy = w1,w2#haveModelPredictValues(step)
			vy=-75
			print(vx)
			print(vy)
			kickBall(pyrep, vx, vy)
			c_data = c_data + [vx, vy]

		state = stepEnvironment(pyrep)
		
		'''
		if i == 25:
			if vx < 0:
				predict = "go_left"
			else:
				predict = "go_right"
			moveGoalKeeper(pyrep,predict)
		'''

		if i > 10: # Force it to make at least 10 steps before checking if it is done
			run = epochFinished(state)
			if run:
				print("Goal detected: Stopping run")
				c_data.append(1.0)
				#rew=1
				break
			if i == max_steps - 1:
				print("Maximum steps exceeded: Stopping run")
				c_data.append(0.0)
				#rew=0
	#print(c_data[len(c_data)-1])
	#Q[Rx.index(vx)]=get_value(c_data[len(c_data)-1],Q[Rx.index(vx)])
	buffer.record((prev, [vx,vy], c_data[-1], state["ball_pos"]))
	if(step>=5):
		buffer.learn(target_actor,target_critic,actor_model,critic_model,gamma,critic_optimizer,actor_optimizer)
		update_target(target_actor.variables, actor_model.variables, tau)
		update_target(target_critic.variables, critic_model.variables, tau)
	pyrep.stop()
	training_data.append(c_data)
pyrep.shutdown()

training_data = np.asarray(training_data)
print(training_data)
