import numpy as np
from math import sqrt

import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.utils import *


class Environment():
    
    def __init__(self,
                 d=3,
                 N_limit=15):
        '''
        d: Dimension.
        N_limit: The upper bound of the kissing number.
        '''
        
        self.d = d
        self.N_limit = N_limit
        
        self.cur_state = self.get_init_state()
        self.accumulate_reward = 0
        self.step_ct = 1        # The first step is defaultly e_1.
        
    
    def get_init_state(self):
        
        Z = np.zeros((self.N_limit, self.d), dtype=np.float32)
        Z[0, 0] = 1
        
        return Z
    
    
    def step(self,
             action):
        '''
        action: [d,] array.
        '''
        
        assert self.step_ct < self.N_limit, "The step count should be smaller than N_limit."
        
        self.cur_state[self.step_ct] = action
        self.accumulate_reward += 1
        self.step_ct += 1
        if self.is_terminate():
            return True
        return False
        
        
    def is_terminate(self):
        '''
        Calculate the "radius" of the polygon.
        If it is larger than 1, then it has not been terminated.
        '''
        
        pass
    
    
    def is_valid(self):
        '''
        Judge that whether the balls intersect.
        '''
        
        G = np.matmul(self.cur_state, self.cur_state.T)
        np.fill_diagonal(G, 0)
        return (np.logical_and(G>=-1, G<=1/2)).all()
    
    
    def reset(self):
        
        self.cur_state = self.get_init_state()
        self.accumulate_reward = 0
        self.step_ct = 1        
        
        
        
if __name__ == "__main__":
    test_env = Environment()
    test_action = np.array([0., 1., 0.])
    test_env.step(test_action)
    print(test_env.is_valid())
    print(test_env.cur_state)
    test_env.step(test_action)
    print(test_env.is_valid())
    print(test_env.cur_state)