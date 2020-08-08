
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import random


# In[3]:


#define total states
state = 6
#initialize Q table to all zero
Q = np.zeros([state,state],dtype=np.int32)
#initialize R table, which is the reward matrix based on the states and actions
R = np.array([[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,-1],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,100],
              [-1,0,-1,-1,0,100]])


# In[13]:


#define hyper parameter, learning rate
r = 0.8
#define hyper parameter, maximum iteration
epochs = 100


# In[23]:


##training to get a new Q table

#maximum iteration is epochs
for epoch in range(epochs):
    #let the initial state traverse every state
    for initial_state in range(state):
        while(True):
            #Find valid path from current state to next state
            valid_action = [idx for idx in range(state) if R[initial_state,idx] != -1]
            #print(valid_action)
            #Choose next state randomly
            next_state_list = random.sample(valid_action,1)
            next_state = next_state_list[0]
            #print(next_state)
            #Find next valid action for next state
            valid_action_next = [idx for idx in range(state) if R[next_state,idx] != -1]
            #Find max Q for next state
            Max_Q = np.max(Q[next_state, valid_action_next])
            #Calculate and update Q value
            Q[initial_state,next_state] = R[initial_state,next_state] + r * Max_Q
            #if next state is 5, then we reach the final state, which is the end condition
            initial_state = next_state
            if(initial_state==5):
                break
    print('Current epoch is: ',epoch, '.\t', 'Maximum Q value is: ', Q.max())


# In[34]:


##test part

#define start from which state
start_state = 2
print(start_state)
#end search if meet the final state 5
while(True):
    #update the value of next_state based on the maximum value of the next actions in Q table
    next_state = Q[start_state].argmax()
    print(next_state)
    start_state = next_state
    if(start_state == 5):
        break

