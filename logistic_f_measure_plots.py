from numpy import *
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random
import pandas as pd
from fairlearn.datasets import fetch_boston


np.random.seed(123456)

def main():
    eps = np.array([0.01, 0.1, 0.5, 1])
    dp_sgd_means = np.array([0.3926, 0.4063, 0.4094, 0.410])
    lopt_means = np.array([0.4461, 0.4916, 0.5085, 0.5169])

    plt.plot(np.arange(len(eps)), dp_sgd_means, label=r"DP-SGD")
    plt.plot(np.arange(len(eps)), lopt_means, label=r"LOPT$_{\epsilon, \delta}$")
    plt.title(r"F-measure for DP-SGD vs. LOPT$_{\epsilon, \delta}$")
    plt.xlabel("Epsilon", fontsize=20)
    plt.ylabel("F-measure", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("images_logistic/f_measure1.png")
    plt.gcf().clear()

if __name__ == "__main__":
    main()
