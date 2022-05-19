import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd, squared_norm
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


def euclid_norm(X, Y):
    d = (X ** 2 + Y ** 2 - 2 * X * Y).sum()
    #     print(d)
    return d


def norm(x):
    return sqrt(squared_norm(x))


def init_matrix(X, n_components, init=None, eps=1e-6, random_state=None):
    n_samples, n_features = X.shape

    if init is None:
        if n_components < n_features:
            init = 'nndsvd'
        else:
            init = 'random'

    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)

        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)

        np.abs(H, H)
        np.abs(W, W)

        return W, H

    if init == 'nndsvd':
        U, S, V = randomized_svd(X, n_components, random_state=random_state)
        W, H = np.zeros(U.shape), np.zeros(V.shape)

        # The leading singular triplet is non-negative
        # so it can be used as is for initialization.
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

        for j in range(1, n_components):
            x, y = U[:, j], V[j, :]

            # extract positive and negative parts of column vectors
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

            # and their norms
            x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
            x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

            # choose update
            if m_p > m_n:
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n

            lbd = np.sqrt(S[j] * sigma)
            W[:, j] = lbd * u
            H[j, :] = lbd * v

        W[W < eps] = 0
        H[H < eps] = 0
        return W, H


class SNMF(object):
    def __init__(self, rank=2, max_iters=2000, mu=1e-14, eps=1e-6, lamda=1, cstab=1e-9, alpha=0.8, output=False,
                 seed=0):

        self.rank = rank
        self.K_matrix = []
        self.X_train = []
        self.X_trained_feature = []
        self.X_test_feature = []

        self.error = []
        self.reconstrcution_loss = []
        self.classification_loss = []

        self.max_iters_train = max_iters
        self.max_iters_test = int(max_iters / 1.5)

        self.alpha = alpha
        self.mu = mu
        self.lamda = lamda
        self.cstab = cstab
        self.eps=eps

        self.output = output

        if seed != 0:
            self.seed = np.random.seed(seed)
        else:
            self.seed = seed

    def fit(self, Data_matrix, label):

        if (len(set(label))) != 2:
            assert ("this method is only for binary classification problem")

        Y = Data_matrix
        M, N = Y.shape
        K, X = init_matrix(Y, self.rank)
        b = np.ones(M)
        sigma = np.random.uniform(-1, 1, self.rank + 1)

        E_sqr_grad_X = 0
        E_sqr_update_X = 0
        E_sqr_grad_sigma = 0
        E_sqr_update_sigma = 0

        lamda=self.lamda
        mu=self.mu
        eps=self.eps
        alpha=self.alpha
        cstab=self.cstab
        if self.output:
            print("Data Matrix:Y shape n*m: {}".format(Y.shape))
            print("K Matrix shape n*rank: {}".format(K.shape))
            print("X Matrix shape rank*m: {}".format(X.shape))


        for iters in range(self.max_iters_train):
            a = np.dot(Y, X.T)
            W = np.c_[b, a]
            Ws = np.dot(W, sigma)
            P = 1. / (1 + np.exp(-Ws))
            loss1 = euclid_norm(Y, np.dot(K, X)) / 2.

            loss2 = lamda / M * ( \
                        np.sum(np.log(1 + np.exp(np.dot(W, sigma)))) - np.sum(np.dot(np.dot(W, sigma), label)))
            loss = loss1 + loss2
            self.error.append(loss)
            self.reconstrcution_loss.append(loss1)
            self.classification_loss.append(loss2)

            # update K function
            K *= np.dot(Y, X.T) / (np.dot(K, (np.dot(X, X.T) + mu)))
            K = np.where(K > eps, K, 0)

            #        update X ADAdelta
            grad_X = np.dot(K.T, (np.dot(K, X) - Y)) + \
                     lamda / len(label) * np.dot(sigma[1:].reshape(-1, 1), np.dot((P - label).T, Y).reshape(-1, 1).T)
            E_sqr_grad_X = alpha * E_sqr_grad_X + (1 - alpha) * (grad_X * grad_X)
            delta_X = -(np.sqrt(E_sqr_update_X + cstab) / np.sqrt(E_sqr_grad_X + cstab)) * grad_X
            E_sqr_update_X = alpha * E_sqr_update_X + (1 - alpha) * (delta_X * delta_X)
            X = X + delta_X
            X = np.where(X < 0, mu, X)

            # update sigma ADAdelta
            grad_sigma = lamda / M * np.dot(W.T, (P - label).T)  # 31
            E_sqr_grad_sigma = alpha * E_sqr_grad_sigma + (1 - alpha) * (grad_sigma * grad_sigma)
            delta_sigma = -np.sqrt(E_sqr_update_sigma + cstab) / np.sqrt(E_sqr_grad_sigma + cstab) * grad_sigma
            E_sqr_update_sigma = alpha * E_sqr_update_sigma + (1 - alpha) * (delta_sigma * delta_sigma)
            sigma = sigma + delta_sigma

            b = W[:, 0]

        self.K_matrix=K
        self.X_train=X
        self.X_trained_feature=np.dot(Data_matrix,X.T)

        return self

    def transform(self, Data_matrix):

        self.X_test_feature=np.dot(Data_matrix,self.X_train.T)
        return self.X_test_feature
