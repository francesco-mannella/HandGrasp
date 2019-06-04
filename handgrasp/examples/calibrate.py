#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
import handgrasp 
import time
import numpy as np


class LinearPolicy:
    """
    A fake controller chosing random actions
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])

    def step(self, observation, reward, done):
        self.action[3:] += .002
        self.action = np.maximum(0,np.minimum(np.pi*0.3, self.action) )
        return self.action

if __name__ == "__main__":
    
    env = gym.make("HandGrasp-v0")
    env.render("human")
    controller = LinearPolicy(env.action_space)

    stime = 20000
    gap = int(20000/300)

    env.reset()

    observation = env.reset()  
    reward = 0 
    done = False

    for t in range(10000): 
        
        action =  controller.step(observation, reward, done)
        
        # do the movement
        state, r, done, info_ = env.step(action)

        time.sleep(1/200)

    
