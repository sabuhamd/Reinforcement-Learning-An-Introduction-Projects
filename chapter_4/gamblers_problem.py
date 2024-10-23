#Reinforcement Learning: Gambler's Problem
#Using Value Iteration to find optimal policy for betting on a flip of a coin to reach goal of $100
#Problem is inspired by Example 4.3 in textbook Reinforcement Learning: An Introduction by Sutton and Barto

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# goal
Goal = 100

# all states: from 0 to 100 dollars to bet
States = np.arange(Goal +1)

# probability of heads
Head_prob = 0.4


def figure_4_3():
    #state value
    state_value = np.zeros(Goal + 1)
    state_value[Goal] = 1.0

    sweeps_history = []

    #Value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)

        for state in States[1:Goal]:
            # get all possible action for current state
            actions = np.arange(min(state, Goal - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    Head_prob * state_value[state + action] + (1 - Head_prob) * state_value[state - action]
                )
            new_value = np.max(action_returns)
            state_value[state] = new_value

        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break

    # compute optimal policy
    policy = np.zeros(Goal + 1)
    for state in States[1:Goal]:
        actions = np.arange(min(state, Goal - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                Head_prob * state_value[state + action] + (1 - Head_prob) * state_value[state - action])

        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label = 'sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    plt.scatter(States, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')

    plt.savefig('figure_4_3.png')
    plt.close()

if __name__ == '__main__':
    figure_4_3()
