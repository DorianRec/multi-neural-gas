# An implementation of an Multi N-Gas

import math
import random

import numpy as np
import matplotlib.pyplot as plt
from random import randint
from random import randrange


# initialize the centroids
def initialize(M, g, K, f):
    if len(f) != g:
        raise Exception('Restriction: len(f) = g')

    # 0.7 Initialize the center vectors randomly
    neurons_M_K_list = []
    for i in range(M):  # for each N-Gas i
        neurons_K = []  # this is the neurons of N-Gas i
        for j in range(K):  # for each neuron
            neurons_K.append(np.random.rand(g) * np.array(f))
        neurons_M_K_list.append(neurons_K)

    # This is the neurons of M Neural-Gas' with K neurons each.
    # Its a numpy array with shape (M,K,g)
    return np.asarray(neurons_M_K_list)


# For Neural-Gas, the dist function is only the difference between i and j
def dist(i, j):
    return abs(i - j)


def train(X, t, neurons_M_K):
    # 1. Choose a stimulus X and apply it to the Neural-Gas
    # 2. Calculate the responses r_k. and
    # 3. Determine the winner i.

    # calculate all euclidean distances
    euclidian_distances_M_K = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            euclidian_distances_M_K[m][k] = np.linalg.norm(X - neurons_M_K[m][k])

    # Determine the Gas with minimum distance
    min_dist = euclidian_distances_M_K[0][0]
    min_m, min_k = 0, 0
    for m in range(M):
        for k in range(K):
            if euclidian_distances_M_K[m][k] < min_dist:
                min_dist = euclidian_distances_M_K[m][k]
                min_m, min_k = m, k

    # min_m is the winner Gas
    # Sort the distances now
    distances_of_winner_neuron = euclidian_distances_M_K[min_m]
    # This contains at position i the index of the neuron, with the i-th smallest euclidean distance.
    sorted_indices = sorted(range(K), key=lambda k: distances_of_winner_neuron[k])

    # 4. Learning rule, calculate Delta C_k
    Delta_C_j = np.zeros((K, g))
    for k in range(K):
        Delta_C_j[sorted_indices[k]] = eta(t) * h(k, s) * (X - neurons_M_K[min_m][sorted_indices[k]])
        # Delta_C_j[sorted_indices[k]] = 0.1 * h(k, t) * (X - neurons_M_K[min_m][sorted_indices[k]])
    # 5. Update step, C_k += Delta C_k
    neurons_M_K[min_m] += Delta_C_j


# The learning rate
def eta(t):
    return a * (b ** t)


# Neighborhood function: Gaussian
# h(dist) = exp (-1/2 * dist^2 / s^2)
# The task wants a fixed s instead of s(t). So t is not used.
def h(dist, s):
    return np.exp(-0.5 * (dist**2) / (s**2))


def generate_three_circles(amount):
    centres = [(0.2, 0.2), (0.4, 0.8), (0.8, 0.2)]
    radius = [0.1, 0.1, 0.1]
    output = []
    for i in range(amount):
        circle_index = randrange(3)
        circle_r = radius[circle_index]
        circle_x, circle_y = centres[circle_index]

        r = circle_r * math.sqrt(random.random())
        theta = random.random() * 2 * math.pi

        x = circle_x + r * math.cos(theta)
        y = circle_y + r * math.sin(theta)
        output.append([x, y])

    return np.asarray(output)


# Given two points (x1,y1),(x2,y2), we want to
# calculate an exponential function y=ab^x.
# Our points are given by (0,eta_0), (P,eta_end)

# First, we find $b$.
# Let w.l.o.g. x2 >= x1. Then we have y2/y1=ab^x2/ab^x1.
# Thus, we have y2/y1=b^(x2-x1) and we get (x2-x1)-th root of (y2/y1) = b.

# After we know b, and we know y=ab^x, find $a$ by a=y/b^x.
def calculate_exponential_decaying_function(x1, y1, x2, y2):
    if x2 >= x1:
        b = (y2 / y1) ** (1 / (x2 - x1))
    else:
        b = (y1 / y2) ** (1 / (x1 - x2))
    a = y1 / (b ** x1)
    return a, b


def plot_exponential_decaying_function(a, b):
    x = np.linspace(0, P, num=50)
    y = a * (b ** x)

    plt.figure()
    plt.plot(x, y)
    plt.plot(0, eta_0, color='green', marker='x', linestyle='dashed', linewidth=2, markersize=12)
    plt.plot(P, eta_end, color='green', marker='x', linestyle='dashed', linewidth=2, markersize=12)
    plt.text(0, eta_0, '$eta_0$')
    plt.text(P, eta_end, '$eta_{end}$')
    plt.title(f"Exponential decaying function $\eta(t)={a}\cdot{b}^t$.")
    plt.xlabel('$x$')
    plt.ylabel('$f(x)=ab^x$')

    plt.show()


def plot_gaussian_neighborhood_function(s):
    x = np.linspace(0, K, num=50)
    y = h(x, s)

    plt.figure()
    plt.plot(x, y)
    plt.title(f"Gaussian neighborhood function s={s}")
    plt.xlabel('$x$')
    plt.ylabel('$h(dist,s)=e^{-1/2 * dist^2 / s^2}$')
    # Since we have the distances 0.0 ... K-1, these values are interesting
    plt.xlim([0.0, K-1])
    plt.ylim([0.0, 1])

    plt.show()


def plot_2dim_data(x_train):
    plt.scatter(*zip(*x_train))
    plt.title('Training data: Centres (0.2, 0.2), (0.4, 0.8), (0.8, 0.2), radius 0.1')
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.show()


# This function works only for FOUR partner N-Gas, since the colors have to be hard coded.
def plot_2dim_neurons(neurons_M_K, stimulus_count):
    for i in range(M):
        plt.scatter(*zip(*neurons_M_K[i].reshape(K, 2)), color=colors[i])
        plt.legend(range(M))

    plt.title("M=" + str(M) + ', N=' + str(N) + ", g=" + str(g) + ", K=" + str(K) + ", (f_1,..,f_g)=" + str(
        [f1, f2, f3, f4, f5][0:g]) + ", Stimulus= " + str(stimulus_count))
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.show()


if __name__ == '__main__':

    # Set values here.
    M = 4  # Number of partner Neural-Gas'
    if M == 0:  # No idea why M=0 should be allowed. But its the task.
        M = 1
    N = 2  # Dimension of input space (N < 7)
    g = 2  # Dimension of grid G. (g <= 5)
    K = 25  # Number of neurons per Neural-Gas

    eta_0 = 0.5
    eta_end = 0.2

    f1, f2, f3, f4, f5 = 1, 1, 1, 1, 1

    s = 8  # This is the fixed size for the gaussian neighborhood-function

    if N < 1 or 6 < N:
        raise Exception('Restriction: 1 <= N < 7')
    if g < 1 or 5 < g:
        raise Exception('Restriction: 1 <= g <= 5')

    # After M is set, set the colors for each neural-Gas, for plotting it
    colors = []
    for i in range(M):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    # Either Load the data
    #x_train = np.loadtxt('PA-E-train2.txt', comments="#")
    # OR generate data
    # Here, i used three circles with centres (0.2, 0.2), (0.4, 0.8), (0.8, 0.2)
    # with radius 0.1 each.
    x_train = generate_three_circles(5000)
    P = x_train.shape[0]

    # Plot the data (If its 2-dimensional)
    plot_2dim_data(x_train)

    # Calculate proper exponential decaying function
    a, b = calculate_exponential_decaying_function(0, eta_0, P, eta_end)
    plot_exponential_decaying_function(a, b)
    plot_gaussian_neighborhood_function(s)

    # Initialize centres
    neurons_M_K = initialize(M, g, K, [f1, f2, f3, f4, f5][0:g])
    # Plot initialized centres
    plot_2dim_neurons(neurons_M_K, 0)

    # Train
    for p in range(P):
        train(x_train[p], p, neurons_M_K)
    plot_2dim_neurons(neurons_M_K, P)

    # Output neurons
    np.savetxt('PA-E-test.txt', neurons_M_K.reshape((M * K, 2)), fmt='%1.6f')
