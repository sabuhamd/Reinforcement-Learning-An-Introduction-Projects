#Reinforcement Learning: Grid Word
#Using RL to find optimal pathing to points A, B
#Inspired by example 3.5 from the book Reinforcement Learning: An Introduction by Sutton and Barto

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

Grid_Size = 5 #5x5 Grid
#Chose different positions than book for funsies
A_Pos = [2,2] # Center
A_Prime_Pos = [0,4] #Top Right
B_Pos = [0,0] #Top Left
B_Prime_Pos = [4, 2] # Middle Bottom
Discount = 0.9

Actions = [np.array([0,-1]), # Left
           np.array([-1,0]), # Up
           np.array([0,1]),  # Right
           np.array([1,0])]  # Down
Action_symbols = ['←', '↑', '→', '↓']

Action_prob = 0.25

def step(state, action):
    if state == A_Pos:
        return A_Prime_Pos, 10   #Move to A' and gain +10 reward when reaching A
    if state == B_Pos:
        return B_Prime_Pos, 5    #Move to B' and gain +5 reward when reaching B

    next_state = (np.array(state) + action).tolist()
    x,y = next_state
    if x < 0 or x >= Grid_Size or y <0 or y >= Grid_Size:
        reward = -1.0
        next_state = state  #If computer makes a move off the grid, punish with -1 reward and set back to previous location
    else:
        reward = 0
    return next_state, reward

#General grid showing special states and map
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):

        #add special state labels
        if [i,j] == A_Pos:
            val = str(val) + " (A)"
        if [i,j] == A_Prime_Pos:
            val = str(val) + " (A')"
        if [i,j] == B_Pos:
            val = str(val) + " (B)"
        if [i,j] == B_Prime_Pos:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text = val,
                    loc = 'center', facecolor='white')

    # Row and column labels
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

#Grid of optimal values
def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox = [0,0,1,1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    #add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in Actions: #Find values for each state and action
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0],next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]  #Find optimal actions
        val = ''
        for ba in best_actions:
            val +=Action_symbols[ba]
        #add special state labels
        if [i,j] == A_Pos:
            val = str(val) + " (A)"
        if [i,j] == A_Prime_Pos:
            val = str(val) + " (A')"
        if [i,j] == B_Pos:
            val = str(val) + " (B)"
        if [i,j] == B_Prime_Pos:
            val = str(val) + " (B')"

        tb.add_cell(i, j, width, height, text = val,
                    loc = 'center', facecolor='white')

    # Row and column labels
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)
#state_value functions for equiprobably random policy
def figure_1():
    value = np.zeros((Grid_Size,Grid_Size))
    while True:
        #Iterate until convergence
        new_value = np.zeros_like(value)
        for i in range(Grid_Size):
            for j in range(Grid_Size):
                for action in Actions:
                    (next_i, next_j), reward = step([i, j], action)
                    #bellman equation
                    new_value[i, j] += Action_prob * (reward + Discount * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('figure_3_1.png')
            plt.close()

            break
        value = new_value

def figure_2():
    # solve linear system of equations by filling coefficients for each state and respective right side constant to find exact solution
    A = -1 * np.eye(Grid_Size*Grid_Size)
    b = np.zeros(Grid_Size*Grid_Size)
    for i in range(Grid_Size):
        for j in range(Grid_Size):
            s = [i,j] #current state
            index_s = np.ravel_multi_index(s, (Grid_Size, Grid_Size))
            for a in Actions:
                s_, r = step(s,a)
                index_s_ = np.ravel_multi_index(s_, (Grid_Size, Grid_Size))

                A[index_s, index_s_] += Action_prob + Discount
                b[index_s] -= Action_prob * r

    x = np.linalg.solve(A, b)
    draw_image(np.round(x.reshape(Grid_Size,Grid_Size),decimals=2))
    plt.savefig('figure_3_2.png')
    plt.close()

#show all optimal moves of each position in grid
def figure_3():
    value = np.zeros((Grid_Size, Grid_Size))
    while True:
        #iterate until convergence
        new_value = np.zeros_like(value)
        for i in range(Grid_Size):
            for j in range(Grid_Size):
                values = []
                for action in Actions:
                    (next_i, next_j), reward = step([i,j], action)
                    # value iteration
                    values.append(reward+Discount*value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            draw_policy(new_value)
            plt.savefig('figure_3_3.png')
            plt.close()
            break
        value= new_value
if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
