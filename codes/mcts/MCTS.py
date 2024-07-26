import numpy as np
import torch
import math
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os
import copy
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join(".")))
from codes.env import Environment
from codes.net import Net
from codes.utils import *


class Node():
    '''
    A node of MCTS.
    '''
    
    def __init__(self,
                 state,
                 parent,
                 pre_action,
                 pre_action_idx,
                 depth=None):
        #####
        # 这里应该初始化一个节点，
        # 包括Q, N以及女儿父母
        # e.g.:
        # self.Q = 0
        # ......
        #####
        
        self.parent = parent    # parent: A Node instance (or None).
        self.pre_action = pre_action     # pre_action: Action (or None).
        self.pre_action_idx = pre_action_idx
        self.is_leaf = True
        self.state = state      # state: Tensor.
        
        self.actions = []       # A list for actions.
        self.children = []      # A list for nodes.
        self.N = []             # A list for visit counts.
        self.Q = []             # A list for action value.
        self.pi = []            # A list for empirical policy probability.
        self.children_n = 0
        
        if depth is not None:
            self.depth = depth
        else:
            node = self
            depth = 1
            while node.parent:
                depth += 1 
                node = node.parent
            self.depth = depth
        
        
    def expand(self,
               net: Net,
               noise=False,
               network_output=None,
               N_limit=15,
               N_samples=32):
        '''
        Expand this node.
        Return the value of this state.
        '''
        
        #FIXME: Here we can apply a transposition table.
        # 1. Get network output.
        # 1.1. If use network to infer:
        if network_output is None:
            state = self.get_network_input(net)
            
            with torch.no_grad():
                mu, v, sigma = net(state)
            policy = sample_actions(mu.repeat(N_samples, 1), sigma.repeat(N_samples, 1))   # [N, d].
            value = v[0]
            policy = np.array(policy)
            raise NotImplementedError, "We need to perform the transformation of policy."
            del mu, sigma
        
        # 1.2. If we already have network output:
        else:
            value, policy = network_output       
            
        # 1.3. Judge whether terminated.
        # We need to judge for each state.
        
        actions = []
        child_states = []
        ct = 0
        
        for idx, action in enumerate(policy):
            child_state = self.state.copy()
            child_state[self.depth] = action
            if is_valid_state(child_state):
                actions.append(action)
                child_states.append(child_state)
                ct += 1 
        
        pi = [1/ct for _ in range(ct)]
        
        if ct == 0 or self.depth >= N_limit:
            self.is_terminal = True
        
        # 2. Check terminal situation.
        if not self.is_leaf:
            raise Exception("This node has been expanded.")
        self.is_leaf = False
        
        if self.is_terminal:    # Mean the state is terminal. Only propagate.
            node = self
            node.is_leaf = True
            value = 0
            while node.parent is not None:
                action_idx = node.pre_action_idx
                node = node.parent
                node.N[action_idx] += 1
                v = (value + 1 * (self.depth - node.depth))
                node.Q[action_idx] = v / node.N[action_idx] +\
                                    node.Q[action_idx] * (node.N[action_idx] - 1) / node.N[action_idx]   
            
            return         

        
        # 3. Get empirical policy probability.
        
        self.actions = actions
        self.pi = pi
        self.children_n = len(actions)
              
        # 4. Init records.
        self.N = [0 for _ in range(len(actions))]
        self.Q = [0 for _ in range(len(actions))]
        
        # 5. Expand the children nodes.
        for idx, action in enumerate(actions):
            child_state = child_states[idx]
            child_depth = self.depth + 1
            child_node = Node(state=child_state,
                              parent=self,
                              pre_action=action,
                              pre_action_idx=idx,
                              depth=child_depth)
            self.children.append(child_node)
            
        # 6. Backward propagate.
        node = self
        while node.parent is not None:
            action_idx = node.pre_action_idx
            node = node.parent
            node.N[action_idx] += 1
            v = (value + 1 * (self.depth - node.depth))
            node.Q[action_idx] = v / node.N[action_idx] +\
                                 node.Q[action_idx] * (node.N[action_idx] - 1) / node.N[action_idx]
    
    
    def select(self, c=None):
        '''
        Choose the best child.
        Return the chosen node.
        '''
        if self.is_leaf:
            raise Exception("Cannot choose a leaf node.")
        
        if c is None:
            c = 1.25 + math.log((1+19652+sum(self.N)) / 19652)
        
        scores = [self.Q[i] + c * self.pi[i] * math.sqrt(sum(self.N)) / (1 + self.N[i])
                  for i in range(self.children_n)]
        
        return self.children[np.argmax(scores)], scores
    
    
    def get_network_input(self, net):
        # Get state for net evaluation.
        raise NotImplementedError, "Here we should perform the transformation of state."
        return torch.tensor(self.state)[None, ...]
        
    

class MCTS():
    '''
    蒙特卡洛树搜索
    '''
    
    def __init__(self,
                 init_state,
                 simulate_times=400,
                 N_samples=32,
                 N_limit=15,
                 **kwargs):
        '''
        超参数传递
        '''
        
        self.simulate_times = simulate_times
        self.N_limit = N_limit
        self.N_samples = N_samples
        if init_state is not None:
            self.root_node = Node(state=init_state,
                                  parent=None,
                                  pre_action=None,
                                  pre_action_idx=None)

    
    def __call__(self,
                 state,
                 net: Net,
                 log=False,
                 verbose=False):
        '''
        进行一次MCTS
        返回: action, actions, visit_pi
        '''

        assert is_equal(state, self.root_node.state), "State is inconsistent."
        iter_item = range(self.simulate_times) if verbose else tqdm(range(self.simulate_times))
        N_limit = self.N_limit
        for simu in iter_item:
            # Select a leaf node.
            node = self.root_node
            while not node.is_leaf:
                node, scores = node.select()         #FIXME: Need to control the factor c.
            node.expand(net, N_limit=N_limit, N_samples=self.N_samples)
        
        actions = self.root_node.actions
        N = self.root_node.N
        visit_ratio = (np.array(N) / sum(N)).tolist()
        action = actions[np.argmax(visit_ratio)]
        
        if log:
            log_txt = self.log()
            return action, actions, visit_ratio, log_txt
        
        return action, actions, visit_ratio
        
    #########################################################
    
    def move(self,
             action):
        '''
        MCTS向前一步
        '''
        assert not self.root_node.is_leaf, "Cannot move a leaf node."
        
        # Get the action idx.
        action_idx = None
        for idx, child_action in enumerate(self.root_node.actions):
            if is_equal(child_action, action):
                action_idx = idx
                
        # Delete other children and move.
        self.root_node.children = [self.root_node.children[action_idx]]
        self.root_node = self.root_node.children[0]
        
        
    def reset(self,
              state,
              simulate_times=None,
              N_limit=None,
              N_samples=None):
        '''
        Reset MCTS.
        '''
        if simulate_times is not None:
            self.simulate_times = simulate_times
        if N_limit is not None:
            self.N_limit = N_limit
        if N_samples is not None:
            self.N_samples = N_samples
        self.root_node = Node(state=state,
                            parent=None,
                            pre_action=None,
                            pre_action_idx=None)
    
    
    def log(self):
        '''
        Get the log text.
        '''
        node = self.root_node
        state_txt = str(node.state)    # state.
        _, scores = node.select()
        N, Q, scores, children = np.array(node.N), np.array(node.Q), np.array(scores), np.array(node.actions)
        top_k_idx = np.argsort(N)[-5:]
        N, Q, scores, children = N[top_k_idx], Q[top_k_idx], scores[top_k_idx], children[top_k_idx]
        
        N_txt, Q_txt, scores_txt, children_txt = str(N), str(Q), str(scores), str(children)
        
        log_txt = "\n".join(
            ["\nCur state: \n", state_txt,
             "\nDepth: \n", str(node.depth),
             "\nchildren: \n", children_txt,
            "\nscores: \n", scores_txt,
            "\nQ: \n", Q_txt,
            "\nN: \n", N_txt,]
        )     
        
        return log_txt   
    
    
if __name__ == '__main__':
    
    init_state = np.array(
        [[[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]],

       [[0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]])
    root_node = Node(state=init_state,
                     parent=None,
                     pre_action=None,
                     pre_action_idx=None)
    
    from net import Net
    net = Net(N_samples=4)   # For debugging.
    
    ############ Debug for Node ############
    # import pdb; pdb.set_trace()
    # root_node.expand(net)
    # children_node = root_node.select()
    # children_node.expand(net)
    # import pdb; pdb.set_trace()
    
    ############ Debug for MCYS ############
    mcts = MCTS(init_state=init_state,
                simulate_times=20)
    import pdb; pdb.set_trace()
    action, actions, pi = mcts(init_state, net)
    import pdb; pdb.set_trace()
    mcts.move(action)
    state = init_state - action2tensor(action)
    import pdb; pdb.set_trace()
    action, actions, pi = mcts(state, net)
    import pdb; pdb.set_trace()