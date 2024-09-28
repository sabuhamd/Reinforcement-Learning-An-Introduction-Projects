"""
Using Reinforcement Learning to train an AI to play Tic Tac Toe
Some Code below taken/ modified from Shangtong Zhang(zhangshangtong.cpp@gmail.com), Jan Hakenberg(jan.hakenberg@gmail.com), Tian Jun(tianjun.cpp@gmail.com), and Kenta Shimada(hyperkentakun@gmail.com)'s
tic_tac_toe project -->  https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter01/tic_tac_toe.py
"""

import numpy as np
import pickle

Amount_cols = 3
Amount_rows = 3
total_board_size = Amount_cols * Amount_rows

class State:
    def __init__(self):
        # 1 represents an X or the mark of the player with the first move
        # -1 represents an O or the mark of the player with the second move
        # 0 represents a blank space on the board
        self.data = np.zeros((Amount_cols, Amount_rows))
        self.winner = None
        self.hash_val = None
        self.end = None


    # compute the unique hash value for 1 state
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i +1

        return self.hash_val
    #check who wins the game or if it is a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []

        # check rows
        for i in range(Amount_rows):
            results.append(np.sum(self.data[i, :]))
        # check cols
        for i in range(Amount_cols):
            results.append(np.sum(self.data[:, i]))

        #check diagonals
        trace = 0
        trace_reverse = 0
        for i in range(Amount_rows):
            trace += self.data[i, i]
            trace_reverse += self.data[i, Amount_rows - 1 - i]
        results.append(trace)
        results.append(trace_reverse)
        # determine winner
        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end


        #determine tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == total_board_size:
            self.winner = 0
            self.end = True
            return self.end

        # game is still running
        self.end = False
        return self.end


    # place marks in position (i, j)
    def next_state(self, i, j, mark):
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = mark
        return new_state

    # Show Board
    def print_state(self):
        for i in range(Amount_rows):
            print('--------------')
            out = '| '
            for j in range(Amount_cols):
                if self.data[i, j] == 1:
                    token = 'X'
                elif self.data[i, j]== -1:
                    token = 'O'
                else:
                    token = ' '
                out += token + ' | '
            print(out)
        print('--------------')

 # Generate all possible game states
def get_all_states_impl(current_state, current_mark, all_states):
    for i in range(Amount_rows):
        for j in range(Amount_cols):
            if current_state.data[i][j] == 0: #check for empty cell as a valid spot for next move
                new_state = current_state.next_state(i, j, current_mark) # generates next state
                new_hash = new_state.hash()
                if new_hash not in all_states:
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end) #store new state
                    if not is_end:
                        get_all_states_impl(new_state, -current_mark, all_states)


def get_all_states():
    current_mark = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_mark, all_states)
    return all_states

    #every possible board configuration
all_states =  get_all_states()

class Judge:
    #player1: moves first, has mark 1
    #player2: moves second, has mark -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.pl_mark = 1
        self.p2_mark = -1
        self.p1.set_mark(self.pl_mark)
        self.p2.set_mark(self.p2_mark)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):
         while True:
            yield self.p1
            yield self.p2

    def play(self, print_state = False):
        alternator = self.alternate()
        self.reset()
        current_state = State()
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True:
            player = next(alternator)
            i, j, mark = player.act()
            next_state_hash = current_state.next_state(i, j, mark).hash()
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner

# AI player
class Player:
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()
        self.step_size = step_size #step size of updating estimations
        self.epsilon = epsilon #probability of an exploration move
        self.states = []
        self.greedy = []
        self.mark = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_mark(self, mark):
        self.mark = mark
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.mark:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0: # differation between a loss and a tie
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5
    #update the value estimation
    def backup(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):
            state = states[i]
            td_error = self.greedy[i] * (
                self.estimations[states[i +1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error


    # based off of state, select the next action
    def act(self):
        state = self.states[-1]
        next_states = []
        next_positions = []
        for i in range(Amount_rows):
            for j in range(Amount_cols):
                if state.data[i, j] == 0: #find open spaces and log them as a potential next move
                    next_positions.append([i, j])
                    next_states.append(state.next_state(
                        i, j, self.mark).hash()
                    )


        if np.random.rand() < self.epsilon: #exploratory move
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.mark)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions): #optimal move
            values.append((self.estimations[hash_val], pos))

        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]
        action.append(self.mark)
        return action
    # save the policy of the AI
    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.mark == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)
    # Load the policy of the AI
    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.mark == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)

# example of the human interface
# input a number to put a mark
# | q | w | e |
# | a | s | d |
# | z | x | c |
# class for the human player
class HumanPlayer:
    def __init__(self, **kwargs):
        self.mark = None
        self.keys = ['q','w', 'e','a','s','d','z','x','c']
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state
    def set_mark(self, mark):
        self.mark = mark

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // Amount_cols
        j = data % Amount_cols
        return i, j, self.mark


#train the AI model
def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)
    judge = Judge(player1, player2)
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judge.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()
        player2.backup()
        judge.reset()
    player1.save_policy()
    player2.save_policy()


# have each ai play compete in specified # of turns
def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judge = Judge(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judge.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judge.reset()

    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


#play with a human player as player1 and ai opponent as player2
# can test if AI can force tie when someone plays 'perfect' tic tac toe
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judge = Judge(player1, player2)
        player2.load_policy()
        winner = judge.play()
        if winner == player2.mark:
            print("You lose!")
        elif winner == player1.mark:
            print("You win!")
        else:
            print("Tie!")


if __name__ == '__main__':
    train(int(1e4))
    compete(int(1e3))
    play()
