"""
Name:        The Self Learning Quant, Example 1

Author:      Daniel Zakrisson

Created:     30/03/2016
Copyright:   (c) Daniel Zakrisson 2016
Licence:     BSD

Requirements:
Numpy
Pandas
MatplotLib
scikit-learn
Keras, https://keras.io/
backtest.py from the TWP library. Download backtest.py and put in the same folder

/plt create a subfolder in the same directory where plot files will be saved

"""
from __future__ import print_function

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing

import backtest as twp

np.random.seed(1335)  # for reproducibility
np.set_printoptions(precision=5, suppress=True, linewidth=150)


# Load data
# def load_data():
#     price = np.arange(200 / 10.0)  # linearly increasing prices
#     return price

def load_data():
    price = np.sin(np.arange(200) / 30.0)  # sine prices
    return price


# Initialize first state, all items are placed deterministically
def init_state(data):
    """init state"""
    close = data
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)

    # --- Preprocess data
    xdata = np.column_stack((close, diff / abs(close)))
    xdata = np.nan_to_num(xdata)
    scaler = preprocessing.StandardScaler()
    xdata = scaler.fit_transform(xdata)

    state = xdata[0:1, :]
    return state, xdata


# Take Action
def take_action(state, xdata, action, signal, time_step):
    # this should generate a list of trade signals that at evaluation time are fed to the backtester
    # the backtester should get a list of trade signals and a list of price data for the assett

    # make necessary adjustments to state and then return it
    time_step += 1

    # if the current iteration is the last state ("terminal state") then set terminal_state to 1
    if time_step == xdata.shape[0]:
        state = xdata[time_step - 1:time_step, :]
        terminal_state = True
        signal.loc[time_step] = 0
        return state, time_step, signal, terminal_state

    # move the market data window one step forward
    state = xdata[time_step - 1:time_step, :]
    # take action
    if action < -0.05:
        signal.loc[time_step] = 100.0 * action.item()
    elif action > 0.05:
        signal.loc[time_step] = 100.0 * action.item()
    else:
        signal.loc[time_step] = 0.0

    # if action != 0:
    #     if action == 1:
    #         signal.loc[time_step] = 100
    #     elif action == 2:
    #         signal.loc[time_step] = -100
    #     elif action == 3:
    #         signal.loc[time_step] = 0
    terminal_state = False

    return state, time_step, signal, terminal_state


# Get Reward, the reward is returned at the end of an episode
def get_reward(new_state, time_step, action, xdata, signal, terminal_state, epoch=0):
    reward = 0
    signal.fillna(value=0, inplace=True)
    if not terminal_state:
        # get reward for the most current action
        forwardsetep = min(time_step + 5, len(xdata) - 1)
        # print(f"forward_step{forwardsetep}, timestep{time_step}")
        reward = (xdata[forwardsetep][0] - xdata[time_step][0]) * action*10
        # if signal[time_step] != signal[time_step - 1]:
        #     i = 1
        #     while signal[time_step - i] == signal[time_step - 1 - i] and time_step - 1 - i > 0:
        #         i += 1
        # reward = (xdata[time_step - 1, 0] - xdata[time_step - i - 1, 0]) * signal[
        #     time_step - 1] * -100 + i * np.abs(signal[time_step - 1]) / 10.0
        # if signal[time_step] == 0 and signal[time_step - 1] == 0:
        #     reward -= 10

    # calculate the reward for all actions if the last iteration in set
    if terminal_state:
        # run backtest, send list of trade signals and asset data to backtest function
        print("signal", signal)
        bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
        reward = bt.pnl.iloc[-1]

    return reward


def evaluate_Q(eval_data, eval_model):
    # This function is used to evaluate the perofrmance of the system each epoch, without the influence of epsilon and random actions
    signal = pd.Series(index=np.arange(len(eval_data)))
    state, xdata = init_state(eval_data)
    # status = 1
    terminal_state = False
    time_step = 1
    model.eval()
    while not terminal_state:
        # We start in state S
        # Run the Q function on S to get predicted reward values on all the possible actions
        # qval = eval_model.predict(state.reshape(1, 2), batch_size=1)
        qval = model(torch.from_numpy(state.reshape(1, 2)).float())
        # action = (torch.argmax(qval))
        action = qval
        # Take action, observe new state S'
        new_state, time_step, signal, terminal_state = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        eval_reward = get_reward(new_state, time_step, action, xdata, signal, terminal_state, i)
        state = new_state
    return eval_reward


# This neural network is the the Q-function, run it like this:
# model.predict(state.reshape(1,64), batch_size=1)
#
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.optimizers import RMSprop

from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()
# criterion = nn.CrossEntropyLoss()
# criterion = nn.SmoothL1Loss()
criterion = nn.MSELoss(reduction='mean')
optim = torch.optim.SGD(model.parameters(), 0.005)
model.train()


def train(data, target):
    # 初始化时，要清空梯度
    with torch.autograd.set_detect_anomaly(True):
        optim.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward(retain_graph=True)
        optim.step()


# model = Sequential()
# model.add(Dense(4, kernel_initializer='lecun_uniform', input_shape=(2,)))
# model.add(Activation('relu'))
# # model.add(Dropout(0.2)) I'm not using dropout in this example
#
# model.add(Dense(4, kernel_initializer='lecun_uniform'))
# model.add(Activation('relu'))
# # model.add(Dropout(0.2))
#
# model.add(Dense(1, kernel_initializer='lecun_uniform'))
# model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

# rms = RMSprop()
# model.compile(loss='mse', optimizer=rms)

import random, timeit

start_time = timeit.default_timer()

indata = load_data()
epochs = 20
gamma = 0.9  # a high gamma makes a long term reward more valuable
alpha = 0.9
epsilon = 1
learning_progress = []
# stores tuples of (S, A, R, S')
h = 0
signal = pd.Series(index=np.arange(len(indata)))
for i in range(epochs):

    state, xdata = init_state(indata)
    state = torch.from_numpy(state).float()
    # print(f"i {i} xdata{xdata[:5]}")
    status = 1
    terminal = False
    time_step = 1
    update = 0
    # while learning is still in progress
    while not terminal:
        # We start in state S
        # Run the Q function on S to get predicted reward values on all the possible actions
        # qval = model.predict(state.reshape(1, 2), batch_size=1)
        # print(f"state in step", state)
        qval = model(state)
        if (random.random() < epsilon) and i != epochs - 1:  # maybe choose random action if not the last epoch
            # action = np.random.randint(0, 4)  # assumes 4 different actions
            action = torch.rand(1)*2 - 1
        else:  # choose best action from Q(s,a) values
            action = qval
        # Take action, observe new state S'
        new_state, time_step, signal, terminal = take_action(state, xdata, action, signal, time_step)
        # Observe reward
        reward = get_reward(new_state, time_step, action, xdata, signal, terminal, i)
        # print(f"state{state}, qval{qval.data}, action{action}, reward{reward}")
        print(f"state{state}, action{action}, reward{reward}")
        # Get max_Q(S',a)
        # newQ = model.predict(new_state.reshape(1, 2), batch_size=1)
        # maxQ = np.max(newQ)
        # y = torch.zeros((1))
        # y[:] = qval[:]
        # y = torch.from_numpy(qval)
        # y = qval
        if not terminal:  # non-terminal state
            # update = (reward + (gamma * maxQ))
            # update = (1 - alpha) * update + alpha * (reward + (gamma * 0))
            update = reward
        else:  # terminal state (means that it is the last state)
            update = torch.tensor(reward).float()
        # y[0] = update  # target output
        # y[:] = update
        # print("update", update)
        # model.fit(state.reshape(1, 2), y, batch_size=1, epochs=1, verbose=0)
        # train(state, update.reshape(1, 1).detach())
        # train(state, action.reshape(1, 1).detach())
        train(state, state[0][1].detach())
        # state = torch.tensor(new_state).float()
        state = torch.from_numpy(new_state).float().detach()
    eval_reward = evaluate_Q(indata, model)
    print("Epoch #: %s Reward: %f Epsilon: %f" % (i, eval_reward, epsilon))
    learning_progress.append((eval_reward))
    if epsilon > 0.1:
        epsilon -= (1.0 / epochs)

elapsed = np.round(timeit.default_timer() - start_time, decimals=2)
print("Completed in %f" % (elapsed,))

# plot results
bt = twp.Backtest(pd.Series(data=[x[0] for x in xdata]), signal, signalType='shares')
bt.data['delta'] = bt.data['shares'].diff().fillna(0)

print(bt.data)

plt.figure()
bt.plotTrades()
plt.suptitle('epoch' + str(i))
plt.savefig('plt/final_trades' + '.png', bbox_inches='tight', pad_inches=1, dpi=72)  # assumes there is a ./plt dir
plt.close('all')

plt.figure()
plt.subplot(3, 1, 1)
bt.plotTrades()
plt.subplot(3, 1, 2)
bt.pnl.plot(style='x-')
plt.subplot(3, 1, 3)
plt.plot(learning_progress)

plt.show()
