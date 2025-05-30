import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join as pjoin

from scipy.optimize import minimize
from scipy.stats import truncnorm, norm
from lib.calculation import moving_window_mean

class RW:
    def __init__(self, v0=0.5, v1=0.5, beta=5, b=0, alpha=0.2) -> None:
        # initialize the record arrays
        self.choices = np.array([])
        self.rewards = np.array([])

        # initialize the model parameters
        self.v0 = v0
        self.v1 = v1
        self.beta = beta
        self.b = b # initialize with no bias
        self.alpha = alpha


    def update(self, outcome: int, choice: int) -> None:
        if choice == -1:
            self.v0 = self.alpha * (outcome - self.v0) + self.v0
            # self.v0 = self.alpha * (outcome - self.v0) + self.gamma * self.v0
            # self.v1 = self.gamma * self.v1
        else:
            self.v1 = self.alpha * (outcome - self.v1) + self.v1
            # self.v1 = self.alpha * (outcome - self.v1) + self.gamma * self.v1
            # self.v0 = self.gamma * self.v0


    # returns the choice made and the probability of making that choice
    def get_choice(self) -> tuple[int, float]:
        if self.choices.size == 0:
            # if this is the first trial, choose without kappa term
            p_r = 1 / (1 + np.exp(-self.beta * (self.v1 - self.v0) + self.b))
        else:
            # pre_choice = self.choices[-1]
            # p_r = 1 / (1 + np.exp(-self.beta * (self.v1 - self.v0) + self.b + self.kappa * pre_choice))
            p_r = 1 / (1 + np.exp(-self.beta * (self.v1 - self.v0) + self.b))
        choice = -1 if np.random.binomial(1, p_r) == 0 else 1
        self.choices = np.append(self.choices, choice)

        return (choice, p_r)


    def nll(self, parameters, choices_real: np.ndarray, rewards_real: np.ndarray) -> float:
        self.assign_parameters(parameters)
        self.choices = np.array([])
        neg_log_likelihood = 0
        self.v0, self.v1 = 0.5, 0.5

        for i in range(choices_real.size):
            choice = choices_real[i]
            c, p_r = self.get_choice()
            self.choices[-1] = choice

            prob = p_r if choice == 1 else 1 - p_r

            if prob == 0:
                prob += 0.01
            
            neg_log_likelihood += -np.log(prob)

            reward = rewards_real[i]

            self.update(reward, choice)
            
        return neg_log_likelihood
    
    def fit(self, choices_real: np.ndarray, rewards_real: np.ndarray) -> Tuple[List[float], float]:
        # fit the model to the data return the best parameters and 
        # the corresponding negative log likelihood
        # parameters: beta, kappa, b
        # RW_Simple simulation
        fitted_parameters: List[float] = []
        nll_min = np.inf

        for i in range(100):
            x0 = self.sample_parameters()
            # bounds = [(1, 10), (-1, 1), (-5, 5), (0.001, 1), (0.001, 1)]
            bounds = [(0.1, np.inf), (-np.inf, np.inf), (0.001, 1)]

            params = minimize(self.nll, x0=x0, args=(choices_real, rewards_real), method='Nelder-Mead', bounds=bounds)['x']
            if self.nll(params, choices_real, rewards_real) < nll_min:
                fitted_parameters = params
                nll_min = self.nll(params, choices_real, rewards_real)

        return fitted_parameters, nll_min

    # parameters: beta, kappa, b
    def assign_parameters(self, parameters):
        self.beta = parameters[0]
        self.b = parameters[1]
        self.alpha = parameters[2]
        # self.gamma = parameters[3]

    def sample_parameters(self):
        beta = np.random.uniform(5, 15)
        b = np.random.uniform(-5, 5)
        alpha = np.random.uniform(0.1, 1)
        return [beta, b, alpha]
    
    # v_r - v_l
    def get_delta_V(self, parameters, choices_real: np.ndarray, rewards_real: np.ndarray, session='session') -> np.ndarray:
        self.assign_parameters(parameters)
        simulated_choices = np.array([])
        self.choices = np.array([])
        delta_V = np.array([])
        self.v0, self.v1 = 0.5, 0.5
        delta_V = np.append(delta_V, self.v0 - self.v1)

        for i in range(choices_real.size):
            choice = choices_real[i]
            c, prob = self.get_choice()
            simulated_choices = np.append(simulated_choices, c)
            self.choices[-1] = choice
            reward = rewards_real[i]
            self.update(reward, choice)
            delta_V = np.append(delta_V, self.v0 - self.v1)
        
        # smoothen the choices
        simulated_choices = moving_window_mean(simulated_choices, 10)
        choices_real = moving_window_mean(choices_real, 10)
        # # plot simulated choices and real choices, and delta_V
        # plt.plot(simulated_choices, label='simulated')
        # plt.plot(choices_real, label='real')
        # plt.legend()

        # # save the plot in data/relative values
        # plt.savefig(pjoin('data', 'relative_values', f'{session}.png'))
        # plt.close()

        return delta_V
