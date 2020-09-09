#################################
# Your name: Shay Fux
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    # TODO: add your code here
    h_lst = []
    alpha_lst = []
    n = len(X_train)
    Dt = np.array([(1 / n) for i in range(n)])
    for i in range(T):
        epsilont, ht = WL(X_train, y_train, Dt)
        h_lst.append(ht)
        alphat = 0.5 * (np.log((1 - epsilont) / epsilont))
        alpha_lst.append(alphat)
        tmp = np.array([Dt[j] * pow(np.e, (-1) * alphat * y_train[j] * predict(ht, X_train[j])) for j in range(n)])
        Dtsum = sum(tmp)
        Dt = tmp / sum(tmp)
    return h_lst, alpha_lst


##############################################
# You can add more methods here, if needed.
def predict(h, x):
    y, j, theta = h
    if x[j] <= theta:
        return y
    else:
        return (-1 * y)


def create_sample(X, Y):
    S = []
    n = len(X)
    for i in range(n):
        S.append((X[i], Y[i]))
    return S


def WL(X_train, Y_train, D):
    wlp = WL_plus(X_train, Y_train, D)
    wlm = WL_minus(X_train, Y_train, D)
    if wlp[0] <= wlm[0]:
        return wlp
    else:
        return wlm


def WL_plus(X_train, Y_train, D):
    S = [[X_train[i], Y_train[i], D[i]] for i in range(len(X_train))]
    d = len(X_train)
    m = len(S)
    F = np.inf
    J = 0
    O = 0
    for j in range(d):
        sortedS = sorted(S, key=lambda x: x[0][j])
        lastx = sortedS[m - 1][0][j] + 1
        tmp = [sortedS[i][2] for i in range(m) if sortedS[i][1] == 1]
        f = sum(tmp)
        if f < F:
            F = f
            O = sortedS[0][0][j] - 1
            J = j
        for i in range(m):
            f = f - sortedS[i][1] * sortedS[i][2]
            if i != m - 1:
                if (f < F) and sortedS[i][0][j] != sortedS[i + 1][0][j]:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + sortedS[i + 1][0][j])
                    J = j
            if i == m - 1:
                if (f < F) and sortedS[i][0][j] != lastx:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + lastx)
                    J = j
    h = (1, J, O)
    return F, h


def WL_minus(X_train, Y_train, D):  # S = {(xi,yi)}_i=1...n is th data set and D is the distribution
    S = [[X_train[i], Y_train[i], D[i]] for i in range(len(X_train))]
    d = len(X_train)
    m = len(S)
    F = np.inf
    J = 0
    O = 0
    for j in range(d):
        sortedS = sorted(S, key=lambda x: x[0][j])
        lastx = sortedS[m - 1][0][j] + 1
        tmp = [sortedS[i][2] for i in range(m) if sortedS[i][1] == -1]
        f = sum(tmp)
        if f < F:
            F = f
            O = sortedS[0][0][j] - 1
            J = j
        for i in range(m):
            f = f + sortedS[i][1] * sortedS[i][2]
            if i != m - 1:
                if (f < F) and sortedS[i][0][j] != sortedS[i + 1][0][j]:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + sortedS[i + 1][0][j])
                    J = j
            if i == m - 1:
                if (f < F) and sortedS[i][0][j] != lastx:
                    F = f
                    O = 0.5 * (sortedS[i][0][j] + lastx)
                    J = j
    h = (-1, J, O)
    return F, h


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, 80)

    ##############################################
    # You can add more methods here, if needed.

    # sec_a(X_train, y_train, X_test, y_test, hypotheses, alpha_vals)
    # sec_b(hypotheses, vocab)
    # sec_c(X_train, y_train, X_test, y_test, hypotheses, alpha_vals)


def number_errors_iteration_t(X_set, Y_set, hypotheses, alpha_vals, T):
    n = len(X_set)
    error_lst = [0 for i in range(T)]
    for i in range(n):
        for t in range(T):
            predict_t = [alpha_vals[j] * predict(hypotheses[j], X_set[i]) for j in range(t + 1)]
            if np.sign(sum(predict_t)) != Y_set[i]:
                error_lst[t] += 1
    error_lst = np.array(error_lst)
    error_lst = error_lst * (1 / n)
    return error_lst


def expoloss(X_set, Y_set, hypotheses, alpha_vals, T):
    n = len(X_set)
    error_lst = [0 for i in range(T)]
    for i in range(n):
        for t in range(T):
            predict_t = [alpha_vals[j] * predict(hypotheses[j], X_set[i]) for j in range(t + 1)]
            error_lst[t] += pow(np.e, (-1) * Y_set[i] * sum(predict_t))
    error_lst = np.array(error_lst)
    error_lst = error_lst * (1 / n)
    return error_lst


def sec_a(X_train, y_train, X_test, y_test, hypotheses, alpha_vals):
    train_error = number_errors_iteration_t(X_train, y_train, hypotheses, alpha_vals, 80)
    test_error = number_errors_iteration_t(X_test, y_test, hypotheses, alpha_vals, 80)
    sum_hyp = [(i + 1) for i in range(80)]
    # print("train error is : ", train_error)
    # print("test error is : ", test_error)
    plt.plot(sum_hyp, train_error, label="train error")
    plt.plot(sum_hyp, test_error, label="test error")
    plt.legend()
    plt.title("train error and test error")
    plt.xlabel('sum of hypotheses')
    plt.ylabel('error percentage')
    # plt.show()


def sec_b(hypotheses, vocab):
    for i in range(10):
        print("the " + str(i + 1) + "t'h" + " wick classifier Adaboost choose is:", hypotheses[i], "and the word is "
                                                                                                   "classified by is",
              vocab.get(hypotheses[i][1]))


def sec_c(X_train, y_train, X_test, y_test, hypotheses, alpha_vals):
    train_exploss = expoloss(X_train, y_train, hypotheses, alpha_vals, 80)
    test_exploss = expoloss(X_test, y_test, hypotheses, alpha_vals, 80)
    sum_hyp = [(i + 1) for i in range(80)]
    # print("train exploss is : ", train_exploss)
    # print("test exploss is : ", test_exploss)
    plt.plot(sum_hyp, train_exploss, label="train exponential")
    plt.plot(sum_hyp, test_exploss, label="test exponential")
    plt.legend()
    plt.title("train exponential and test exponential")
    plt.xlabel('sum of hypotheses')
    plt.ylabel('exponential loss')
    # plt.show()


##############################################


if __name__ == '__main__':
    main()
