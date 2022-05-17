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

# for all the code below, K = 2

def f1(actual, predicted, label):
    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual==label) & (predicted==label))
    fp = np.sum((actual!=label) & (predicted==label))
    fn = np.sum((predicted!=label) & (actual==label))

    precision = tp/(tp+fp+0.0000000000000000000001)
    recall = tp/(tp+fn+0.0000000000000000000001)
    f1 = 2 * (precision * recall) / (precision + recall + 0.0000000000000000000001)
    return f1

def f_measure(beta, x, y):
    d = len(beta)
    n = len(x)
    probs = None
    if d == 2:
        probs = exp(log_sigmoid(beta[0] + x*beta[1]))
    else:
        probs = exp(log_sigmoid(beta[0] + np.matmul(x, beta[1:])))
    pred = np.round(np.array(probs))

    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(y, pred, label) 
        for label in np.unique(y)])

# calculates prediction error
def error(beta, x, y):
    d = len(beta)
    n = len(x)
    probs = None
    if d == 2:
        probs = exp(log_sigmoid(beta[0] + x*beta[1]))
    else:
        probs = exp(log_sigmoid(beta[0] + np.matmul(x, beta[1:])))
    pred = np.round(np.array(probs))
    return np.sum(np.abs(pred - y))/n

# calculates prediction error for each group
def error_all_groups(beta, x1, x2, y1, y2):
    n1 = len(x1)
    n2 = len(x2)

    d = len(beta)
    error1 = 0
    error2 = 0
    probability = None
    if d == 2:
        probs = exp(log_sigmoid(beta[0] + x1*beta[1]))
        pred = np.round(np.array(probs))
        error1 = np.sum(np.abs(pred - y1))/n1

        probs = exp(log_sigmoid(beta[0] + x2*beta[1]))
        pred = np.round(np.array(probs))
        error2 = np.sum(np.abs(pred - y2))/n2
    else:
        probs = exp(log_sigmoid(beta[0] + np.matmul(x1, beta[1:])))
        pred = np.round(np.array(probs))
        error1 = np.sum(np.abs(pred - y1))/n1

        probs =  exp(log_sigmoid(beta[0] + np.matmul(x2, beta[1:])))
        pred = np.round(np.array(probs))
        error2 = np.sum(np.abs(pred - y2))/n2

    return np.array([error1, error2])

def gaussian(shift=0., scale=1., size=None):
    """Sample from the Gaussian distribution."""
    draws = np.random.normal(loc=shift, scale=scale, size=size)
    return draws

def log_sigmoid(x):
    return log(1.0)-log(1.0 + exp(-x))

def sigmoid(x):
    return 1.0/(1.0 + exp(-x))

def logit_f1(beta, data):
    x, y = get_x_y_1(data)
    d = len(beta)
    n = len(x)
    probs = None
    if d == 2:
        probs = exp(log_sigmoid(beta[0] + x*beta[1]))
    else:
        probs = exp(log_sigmoid(beta[0] + np.matmul(x, beta[1:])))

    label = 0
    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((y*probs))
    fp = np.sum((1-y)*probs)
    fn = np.sum(y*(1-probs))

    precision = tp/(tp+fp+0.0000000000000000000001)
    recall = tp/(tp+fn+0.0000000000000000000001)
    f1 = 2 * (precision * recall) / (precision + recall+0.0000000000000000000001)
    return -f1

# Here is the likelihood function for a logit
def logit_neg_log_likelihood(beta, data):
    """negative log likelihood function for a logit
    :param beta: 1darray of [intercept, slope]
    """
    x, y = get_x_y_1(data)
    d = len(beta)
    probability = None
    # Here is the systematic component
    if d == 2:
        probability = exp(log_sigmoid(beta[0] + x*beta[1]))
    else:
        probability = exp(log_sigmoid(beta[0] + np.matmul(x, beta[1:])))

    # Here is the stochastic component
    log_likelihood = y * log(probability + 0.0000000000000000000001) + (1 - y) * log(1 - probability + 0.0000000000000000000001)

    return -log_likelihood

# Calculate the gradient at a point in the parameter space
def calc_clamped_gradient(X, C, d, beta, fun):
    """
    :param X: data
    :param C: clipping norm bound
    :param d: dimension
    :param beta: learned param
    :param fun: loss function to optimize
    """

    dx = 0.0001
    bounds = [-C, C]

    if d==2:
        out = fun(beta=beta, data=X)
        out1 = fun(beta=beta + [dx, 0], data=X) # intercept
        out2 = fun(beta=beta + [0, dx], data=X)

        Del_1 = (out1-out) / dx 
        Del_1_clamped = np.clip(Del_1, *bounds)    
        mean_Del_1 = Del_1_clamped.mean()

        Del_2 = (out2-out) / dx
        Del_2_clamped = np.clip(Del_2, *bounds) 
        mean_Del_2 = Del_2_clamped.mean()

        return np.array([mean_Del_1, mean_Del_2])
    elif d==3:
        out = fun(beta=beta, data=X)
        out1 = fun(beta=beta + [dx, 0, 0], data=X)
        out2 = fun(beta=beta + [0, dx, 0], data=X)
        out3 = fun(beta=beta + [0, 0, dx], data=X)

        Del_1 = (out1-out) / dx 
        Del_1_clamped = np.clip(Del_1, *bounds)    
        mean_Del_1 = Del_1_clamped.mean()

        Del_2 = (out2-out) / dx
        Del_2_clamped = np.clip(Del_2, *bounds) 
        mean_Del_2 = Del_2_clamped.mean()

        Del_3 = (out2-out) / dx
        Del_3_clamped = np.clip(Del_3, *bounds) 
        mean_Del_3 = Del_3_clamped.mean()

        return np.array([mean_Del_1, mean_Del_2, mean_Del_3])
    else:
        return None


# xs -> columns of features to extract
# y  -> columns of target variable
def read_file_by_columns(xs, y, target_text, fname):
    data = pd.read_csv(fname, header=None)
    Xarr = data[xs].values
    yarr = np.array(list(map(lambda x: 1 if x==target_text else 0, data[y].values)))

    model = LogisticRegression().fit(X = Xarr, y=yarr)
    data = np.hstack((data[xs].values, yarr[:,None]))

    N = len(Xarr)
    L = round(sqrt(N))     # This is the recommended batch size

    steps = np.ceil(N / L)   # Happens to be ~L 

    epsilon = 1
    delta = 1e-6

    dp_sgd(data, epsilon, delta, steps, L, model, 'ADULT')
    
def dp_sgd(data, eps, delta, steps, L):
    ## Shuffle the data
    np.random.shuffle(data)

    d = 2
    C = 10                      # Interval to clip over
    beta = np.zeros(d)         # Starting parameters
    nu = np.ones(d)             # Learning speeds

    history = [beta]

    t = 1
    # Run one epoch of SGD
    for batch in np.array_split(data, steps):
        sensitive_grad = calc_clamped_gradient(X=batch, C=C, d=d, beta=beta, fun=logit_neg_log_likelihood)

        sensitivity = 2 * C / L  # 2 * C / len(batch)
        functional_epsilon = eps * L/2 # epsilon * sqrt(steps) 
        scale = (sensitivity / functional_epsilon) * np.sqrt(2*np.log(2/delta))

        Del = sensitive_grad + gaussian(shift=0, scale=scale, size=d)
        #cat("Del:  ",Del,"\n")
        beta = beta - Del * nu#/math.sqrt(t)
        t += 1
        #cat("Beta:",beta, "\n")

        history.append(beta)

    history = np.array(history)

    '''
    plt.plot(np.arange(len(history)), history[:, 0], label=r"DP-SGD for $\hat{\beta}_0$")
    plt.title(r"DP-SGD convergence for $\beta_0$ on {0} dataset".format(text))
    plt.axhline(y=model.intercept_[0], color="red", linestyle="--", label=r"Non-private $\hat{\beta}_0$")
    plt.xlabel("Number of Gradient Updates", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("images_logistic/beta0_{0}.png".format(text))
    plt.gcf().clear()

    plt.plot(np.arange(len(history)), history[:, 1], label=r"DP-SGD for $\hat{\beta}_1$")
    plt.title(r"DP-SGD convergence for $\beta_1$ on {0} dataset".format(text))
    plt.axhline(y=model.coef_[0, 0], color="red", linestyle="--", label=r"Non-private $\hat{\beta}_1$")
    plt.xlabel("Number of Gradient Updates", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig("images_logistic/beta1_{0}.png".format(text))
    plt.gcf().clear()
    '''

    return beta

def main1():
    read_file_by_columns([0, 4], 14, ' >50K', '~/Downloads/adult.data')

    csv_data = pd.read_csv(
        "https://raw.githubusercontent.com/privacytoolsproject/cs208/master/data/MaPUMS5full.csv")

    # define public information
    n = len(csv_data)            # in this case, dataset length is considered public, and is not protected
    data = csv_data[['educ', 'employed']].values.astype(float)
    d = 2

    dis = data[:, 0]
    married = data[:, 1]
    model = LogisticRegression().fit(X = dis[:, None], y=married)

    N = len(dis)
    L = round(sqrt(N))     # This is the recommended batch size

    steps = np.ceil(N / L)   # Happens to be ~L
    # per element per step
    epsilon = 1
    delta = 1e-6

    beta = dp_sgd(data, epsilon, delta, steps, L, model, "PUMS")

    data = csv_data[['educ', 'black', 'employed']].values.astype(float)
    x1, x2, y1, y2 = get_x_y_2(data)
    print(beta)
    print("Error w/o fairness: ", error_all_groups(beta, x1, x2, y1, y2))

# calculates the gradient of l(c, K, data) with differential privacy and standard errors
def pgrad_l_se(beta, data, eps, delta, splits):
    d = 2
    K = 2
    vec = np.zeros(K*d).reshape(K, d)
    C = 10

    L = len(data)

    # group 1
    sensitivity1 = 2 * C / L  # 2 * C / len(batch)
    scale1 = (sensitivity1 / eps) * np.sqrt(2*np.log(2/delta))
    grad0 = calc_clamped_gradient(X=data, C=C, d=d, beta=beta, fun=logit_neg_log_likelihood)
    vec[0] = grad0 + gaussian(shift=0, scale=scale1*splits[0], size=d)

    # group 2
    # TODO: use gradient of F-measure
    sensitivity2 = 2 * C / L  # 2 * C / len(batch)
    scale2 = (sensitivity2 / eps) * np.sqrt(2*np.log(2/delta))
    grad1 = calc_clamped_gradient(X=data, C=C, d=d, beta=beta, fun=logit_f1)
    vec[1] = grad1 + gaussian(shift=0, scale=scale2*splits[1], size=d)

    return vec.transpose()

# private version of computing losses that uses standard errors
def pl_se(beta, data, eps, delta, splits):
    K = 2
    vec = np.zeros(K)
    C = 10
    sensitivity = 1  # 2 * C / len(batch)

    scale = (sensitivity / eps) * np.sqrt(2*np.log(2/delta))

    x, y = get_x_y_1(data)
    error0, error1 = error(beta, x, y), 1-f_measure(beta, x, y)
    vec[0] = error0 + gaussian(shift=0, scale=scale*splits[0], size=1)
    vec[1] = error1 + gaussian(shift=0, scale=scale*splits[0], size=1)

    return vec

# calculates the differentially private gradient of g on the dataset
# assume smallest group is second group (indexed by 1)
def pgrad_g(beta, vec):
    K = len(vec)
    if K == 2:
        vec_l = vec
        big_vec = np.zeros(len(vec))
        big_vec[1] = 1
        return big_vec
    else:
        return grad_smooth_max(vec, 1)

# calculates the smooth maximum
def smooth_max(vec, eta):
    vec = np.array(vec)
    den = np.sum(np.exp(eta*vec))
    return np.sum(vec*np.exp(eta*vec))/den

# calculates the gradient of smooth maximum
def grad_smooth_max(vec, eta):
    return np.exp(eta*vec)/np.sum(np.exp(eta*vec)) * (1 + eta*(vec - smooth_max(vec, eta)))

# calculates the value of g on the dataset
# assume smallest group is second group (indexed by 1)
def g(beta, data):
    x, y = get_x_y_1(data)
    return f_measure(beta, x, y)

def get_x_y_2(data):
    data_non_b = data[data[:,1]==0]
    data_b = data[data[:,1]==1]
    x1 = data_non_b[:,:-2]
    y1 = data_non_b[:,-1]
    x2 = data_b[:,:-2]
    y2 = data_b[:,-1]
    y1 = y1[:,None]
    y2 = y2[:,None]

    return x1, x2, y1, y2

def get_x_y_1(data):
    x1, x2, y1, y2 = get_x_y_2(data)
    return np.concatenate((x1, x2)), np.concatenate((y1, y2))

# Calculates loss for each group
# returns vec which is loss for each group.
def l(c, x, y):
    return [error(beta, x, y), 1-f_measure(beta, x, y)]

# calculates the gradient of f on the dataset with differential privacy
def pgrad_f(theta, vec):
    K = len(vec)
    return np.ones(K)

def dp_sgd_lopt(data, epsilon, delta, steps, L):
    ## Shuffle the data
    np.random.shuffle(data)

    d = 2
    C = 10                      # Interval to clip over
    beta = np.zeros(d)          # Starting parameters
    nu = np.ones(d)             # Learning speeds

    history = [beta]

    splits = np.array([1, 1])

    l_f = 2
    K = 2
    alpha = 1
    G = (alpha + l_f*math.sqrt(K))/alpha
    nu = np.array([1, 0.01])

    t = 1
    # Run one epoch of SGD
    for batch in np.array_split(data, steps):
        f_eps = epsilon * L/2 # epsilon * sqrt(steps)

        private_grad_l = pgrad_l_se(beta, batch, f_eps/2, delta/2, splits)
        private_l = pl_se(beta, batch, f_eps/2, delta/2, splits)

        Grad_g = 0 if g(beta, batch) <= 0 else np.matmul(private_grad_l, pgrad_g(beta, private_l))
        Del = np.matmul(private_grad_l, pgrad_f(beta, private_l)) + G*Grad_g

        #cat("Del:  ",Del,"\n")
        beta = beta - Del * nu/math.sqrt(t)
        t += 1

        #cat("Beta:",beta, "\n")

        history.append(beta)

    '''
    print(error_history.shape)
    print(error_history)
    plt.plot(np.arange(len(error_history)), error_history[:, 0], label=r"DP-SGD for $\hat{\beta}_0$")
    #plt.title(r"DP-SGD convergence for $\beta_0$ on {0} dataset".format(text))
    plt.xlabel("Number of Gradient Updates", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
              
    plt.plot(np.arange(len(error_history)), error_history[:, 1], label=r"DP-SGD for $\hat{\beta}_0$")
    #plt.title(r"DP-SGD convergence for $\beta_0$ on {0} dataset".format(text))
    plt.xlabel("Number of Gradient Updates", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.legend(fontsize=15)
    plt.show()
    '''

    return beta

def main2():
    csv_data = pd.read_csv(
        "https://raw.githubusercontent.com/privacytoolsproject/cs208/master/data/MaPUMS5full.csv")
    n = len(csv_data)
    data = csv_data[['educ', 'black', 'employed']].values.astype(float)
    d = 2

    L = round(sqrt(n))     # This is the recommended batch size

    steps = np.ceil(n / L)   # Happens to be ~L
    # per element per step
    delta = 1e-6

    x, y = get_x_y_1(data)
    f_sgd = []
    f_lopt = []

    for epsilon in [0.01, 0.1, 0.5, 1]:
        print("Epsilon: ", epsilon)

        for i in range(10):
            print("Iteration {0}".format(i))
            beta1 = dp_sgd(data, epsilon, delta, steps, L)
            #beta2 = dp_sgd_lopt(data, epsilon, delta, steps, L)
            beta2 = dp_sgd_lopt(data, epsilon, delta, n/10, L)

            f_sgd.append(f_measure(beta1, x, y))
            f_lopt.append(f_measure(beta2, x, y))

        print("F-measure for DP-SGD (Mean:{0}) (Std:{1})".format(np.mean(f_sgd, axis=0), np.std(f_sgd, axis=0)))
        print("F-measure for LOPT (Mean:{0}) (Std:{1})".format(np.mean(f_lopt, axis=0), np.std(f_lopt, axis=0)))

def main3():
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 80)
    X, y = fetch_boston(as_frame=True, return_X_y=True)
    boston_housing=pd.concat([X, y], axis=1)
    data = boston_housing[['CRIM', 'CHAS']].values.astype(float)
    d = 2

    dis = data[:, 0]
    married = data[:, 1]
    model = LogisticRegression().fit(X = dis[:, None], y=married)

    N = len(dis)
    L = round(sqrt(N))     # This is the recommended batch size

    steps = np.ceil(N / L)   # Happens to be ~L
    # per element per step
    epsilon = 1
    delta = 1e-6
    print("steps:", steps)
    print("N: ", N)

    dp_sgd(data, epsilon, delta, steps, L, model, "BOSTON")

if __name__ == "__main__":
    #main1()

    main2()

    #main3()

