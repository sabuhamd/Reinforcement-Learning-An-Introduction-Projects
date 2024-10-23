#Reinforcement Learning: Car Rental
#Using Policy Iteration to find optimal policy to move cars back and forth between 2 rental locations
#Inspired by Example 4.2 in Reinforcement Learning: An Introduction by Sutton and Barto. More details on the problem there

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

matplotlib.use('Agg')

#maximum number of cars at each location
Max_Cars = 20

#maximum number of cars available to move each night
Max_Move = 5

#expectaion of rental requests for first location
Rent_Req_First = 3

#expectation of rental requests for second location
Rent_Req_Second = 4

# expectation for number of cars returned in first location
Returns_First = 3

#expectation for number of cars returned in second location
Returns_Second = 2

Discount = 0.9

#money made by renting a car
Rental_Mulah = 10

#cost of moving a car
Move_Car_Cost = 2

# all possible actions
actions = np.arange(-Max_Move, Max_Move +1)

# Upper bound for poission distribution
#If n is greater than upper bound, then the probability of getting n is truncated to 0
Poisson_Upp_Bound = 11

#Probability for poisson distribution
#@lam: lambda should be less than 10 for our function
poisson_cache = dict()

def poisson_probability(n, lam):
    global poission_cache
    key = n * 10 + lam #generate unique key for specific n and lambda
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n,lam) #calculate poisson pmf for n and lambda
    return poisson_cache[key]

def expected_return(state, action, state_value, constant_returned_cars):

    #@state: [number of cars in first location, number of cars in second location]
    #@action: positive if moving cars from first location to second location,
    #         negative if moving from second to first
    #@statevalue: state value matrix
    #@constant_returned_cars: if set True, model is simplified such that the number of cars
    #returned in daytime becomes constant rather than a random value from poisson distribution

    #initialize total return
    returns = 0.0

    #factor in cost of moving cars
    returns -= Move_Car_Cost * abs(action)

    # finding number of cars at each location by factoring in moving cars
    Num_Cars_First = min(state[0] - action, Max_Cars)
    Num_Cars_Second = min(state[1] + action, Max_Cars)

    # run through each possible rental request
    for rent_req_first in range(Poisson_Upp_Bound):
        for rent_req_second in range(Poisson_Upp_Bound):
            #probability for current combination of rental requests
            prob = poisson_probability(rent_req_first, Rent_Req_First) * \
                poisson_probability(rent_req_second, Rent_Req_Second)

            num_cars_first = Num_Cars_First
            num_cars_second = Num_Cars_Second

            # valid rent requests should be less than number of cars in specified location
            valid_rent_first = min(num_cars_first, rent_req_first)
            valid_rent_second = min(num_cars_second, rent_req_second)

            # reward system for renting out cars
            reward = (valid_rent_first + valid_rent_second) * Rental_Mulah
            num_cars_first -= valid_rent_first
            num_cars_second -= valid_rent_second

            if constant_returned_cars:
                # get returned cars, then those cars can be used for rentals the next day
                returned_cars_first = Returns_First
                returned_cars_second = Returns_Second
                num_cars_first = min(num_cars_first + returned_cars_first, Max_Cars)
                num_cars_second = min(num_cars_second + returned_cars_second, Max_Cars)
                returns += prob * (reward + Discount * state_value[num_cars_first, num_cars_second])
            else:
                for returned_cars_first in range(Poisson_Upp_Bound):
                    for returned_cars_second in range(Poisson_Upp_Bound):
                        prob_return = poisson_probability(
                            returned_cars_first, Returns_First) * poisson_probability(returned_cars_second, Returns_Second)
                        num_cars_first_ = min(num_cars_first+ returned_cars_first, Max_Cars)
                        num_cars_second_ = min(num_cars_second, returned_cars_second, Max_Cars)
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + Discount * state_value[num_cars_first_, num_cars_second_])
    return returns

def figure_4_2(constant_returned_cars=True):
    value = np.zeros((Max_Cars + 1, Max_Cars + 1))
    policy = np.zeros(value.shape, dtype = np.int64)

    iterations = 0
    _, axes = plt.subplots(2, 3, figsize=(40,20))
    plt.subplots_adjust(wspace=0.1, hspace = 0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax = axes[iterations])
        fig.set_ylabel('Num of cars at first location', fontsize = 30)
        fig.set_yticks(list(reversed(range(Max_Cars+1))))
        fig.set_xlabel('Num of cars at second location', fontsize = 30)
        fig.set_title('policy {}'.format(iterations), fontsize=30)

        # policy evaluation
        while True:
            old_value = value.copy()
            for i in range(Max_Cars+1):
                for j in range(Max_Cars+1):
                    new_state_value = expected_return([i, j], policy[i, j], value, constant_returned_cars)
                    value[i, j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if max_value_change < 1e-4:
                break

        # policy improvement
        policy_stable = True
        for i in range(Max_Cars + 1):
            for j in range(Max_Cars + 1):
                old_action = policy[i, j]
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(expected_return([i, j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)
                new_action = actions[np.argmax(action_returns)]
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:
                    policy_stable = False
        print('policy stable {}'.format(policy_stable))

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax = axes[-1])
            fig.set_ylabel('Num of cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(Max_Cars+1))))
            fig.set_xlabel('Num of cars at second location', fontsize = 30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig("figure_4_2.png")
    plt.close()

if __name__ == '__main__':
    figure_4_2()
