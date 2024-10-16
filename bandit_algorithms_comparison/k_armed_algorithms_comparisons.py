"""
Comparing the efficiency of different k armed bandit algorithms

"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')

class Bandit:
    def __init__(self, k_arm = 10, epsilon=0., initial=0, step_size = 0.1, sample_averages=False, UCB_param = None,
                 gradient = False, gradient_baseline = False, true_reward = 0.):
        self.k = k_arm # of arms
        self.step_size = step_size # constant step size for updating estimations
        self.sample_averages = sample_averages # if true, use sample averages to update estimations instead of constant step size
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param # if not None, use UCB algortihm to selct action
        self.gradient = gradient # if true, use gradient based bandit algorithm
        self.gradient_baseline = gradient_baseline # if true use average reward as baseline for gradient based bandit algorithm
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon # probability for exploration in epsilon-greedy algorithm
        self.initial = initial # initial estimation for each action

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # number of times each action is chosen
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get action for a particular bandit
    def act(self):
        if np.random.rand() < self.epsilon: # exploration move in epsilon greedy algorithm
            return np.random.choice(self.indices)

        if self.UCB_param is not None: # using UCB algorithm to select action
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5 ))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient: # use gradient based algorithm to select actions
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p = self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])


    # take action, update estimation for the action
    def step(self, action):
        # generate the reward with N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            #update estimation with sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            # update estimation using stochastic gradient descent
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                # add gradient baseline to estimation update equation
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

# run the k armed bandit algorithms and record the amount of times the algorithm chooses an optimal action and rewards accumalated
def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

#plotting reward distribution of actions
def figure_1():
    plt.violinplot(dataset = np.random.randn(200,10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('figure_2_1.png')
    plt.close()

#plot showing reward distribution of various epsilon greedy bandit algorithms
def figure_2(runs = 2000, time = 1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10,20))

    plt.subplot(2,1,1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label = '$\epsilon = %.02f$' % (eps))

    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2,1,2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label ='$\epsilon = %.02f$' % (eps) )
    plt.xlabel('steps')
    plt.ylabel('% Optimal action')
    plt.savefig('figure_2_2.png')
    plt.close()


# plotting  different levels of epsilon values and initial estimates
def figure_3(runs = 2000, time = 1000):
    bandits = []
    bandits.append(Bandit(epsilon = 0, initial = 5, step_size=0.1))
    bandits.append(Bandit(epsilon = 0.1, initial = 0, step_size=0.1))
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label = '$\epsilon = 0, q = 5$' )
    plt.plot(best_action_counts[1], label = '$\epsilon = 0.1, q = 0$')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig('figure_2_3.png')
    plt.close()

#plot comparison of UCB algorithm and epsilon greedy algorithm
def figure_4(runs=2000, time = 1000):
    bandits = []
    bandits.append(Bandit(epsilon = 0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon = 0.1, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label ='UCB $c = 2$' )
    plt.plot(average_rewards[1], label = 'epsilon greedy $\epsilon = 0.1$')
    plt.xlabel('Steps')
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('figure_2_4.png')
    plt.close()


# plot comparing diffirent steps sizes and gradient baselines
def figure_5(runs = 2000, time = 1000):
    bandits =[]
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label = labels[i])
    plt.xlabel('Steps')
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('figure_2_5.png')
    plt.close()

#plot comparison of 4 different models: epsilon greedy, gradient, UCB, optimistic initialization
def figure_6(runs = 2000, time = 1000):

     labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
     generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                   lambda alpha: Bandit(gradient=True, step_size=alpha, gradient_baseline=True),
                   lambda coef: Bandit(epsilon=0, UCB_param=coef, sample_averages=True),
                   lambda initial: Bandit(epsilon=0, initial=initial, step_size=0.1)]
     parameters = [np.arange(-7, -1, dtype=np.float64),
                   np.arange(-5, 2, dtype=np.float64),
                   np.arange(-4, 3, dtype=np.float64),
                   np.arange(-2, 3, dtype=np.float64)
                   ]
     bandits = []
     for generator, parameter in zip(generators, parameters):
         for param in parameter:
             bandits.append(generator(pow(2, param)))

     _, average_rewards = simulate(runs, time, bandits)
     rewards = np.mean(average_rewards, axis = 1)


     i = 0
     for label, parameter in zip(labels, parameters):
         l = len(parameter)
         plt.plot(parameter, rewards[i:i+l], label = label)
         i +=1
     plt.xlabel('Parameter($2^x$)')
     plt.ylabel('Average reward')
     plt.legend()
     plt.savefig('figure_2_6.png')
     plt.close()


if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
