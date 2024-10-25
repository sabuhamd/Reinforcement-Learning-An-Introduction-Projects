#Reinforcement Learning: Grid_world 2
#Problem is inspired by example 4.1 from textbook Reinforcement Learning: An Introduction by Sutton and Barto

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

Grid_Size = 4

Actions = [np.array([0, -1]), # left
           np.array([-1, 0]), # up
           np.array([0, 1]),  # right
           np.array([1, 0])]  # down
Action_Prob = 0.25

def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == Grid_Size - 1 and y == Grid_Size - 1)

#step function to find states when computer moves on grid
def step(state, action):
    if is_terminal(state):
        return state, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    if x < 0 or x >= Grid_Size or y < 0 or y>= Grid_Size: # if computer goes off grid,punish with -1 reward
        next_state = state

    reward = -1
    return next_state, reward

# Draw gird world visual
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox = [0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text = val,
                    loc = 'center', facecolor = 'white')

        # Add row and column labels
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text = i+1, loc = 'right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)

# compute state values for each state
def compute_state_value(in_place=True, discount=1.0):
    new_state_values = np.zeros((Grid_Size, Grid_Size))
    iteration = 0
    while True:
        if in_place:
            state_values = new_state_values
        else:
            state_values = new_state_values.copy()
        old_state_values = state_values.copy()

        for i in range(Grid_Size):
            for j in range(Grid_Size):
                value = 0
                for action in Actions:
                    (next_i, next_j), reward = step([i, j], action)
                    value += Action_Prob * (reward + discount * state_values[next_i, next_j])
                new_state_values[i, j] = value

        max_delta_value = abs(old_state_values - new_state_values).max()
        #converge to optimal state values
        if max_delta_value < 1e-4:
            break

        iteration += 1

    return new_state_values, iteration

def figure_4_1():

    _, async_iteration = compute_state_value(in_place=True)
    values, sync_iteration = compute_state_value(in_place=False)
    draw_image(np.round(values, decimals=2))
    print('In-place: {} iterations'.format(async_iteration)) #use in place policy evaluation
    print('Synchronous: {} iterations'.format(sync_iteration)) #use out of place policy evaluation

    plt.savefig('figure_4_1.png')
    plt.close()

if __name__ == '__main__':
    figure_4_1()
