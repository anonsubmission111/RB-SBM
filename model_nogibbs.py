import numpy as np
from scipy.special import digamma


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1, take_exp=True):
    if take_exp:
        x = x - np.expand_dims(x.max(axis=axis), axis=axis).repeat(x.shape[axis], axis)
        x = np.exp(x)
    return x / np.expand_dims(x.sum(axis), axis=axis).repeat(x.shape[axis], axis)


def xavier_init(shape0, shape1):
    return np.random.normal(loc=0, scale=2/(shape0 + shape1), size=(shape0, shape1))


class RBMSBM(object):
    """
    Implements the model that combines RBM and SBM for directed networks with a prior on B. Adds exact calculation of
    partition function in the M-step.
    """
    def __init__(self, N, K, M, alphas, betas):
        """
        :param N: Number of nodes
        :param K: Number of communities
        :param M: Number of attributes
        :param alphas: (K, K) matrix of prior alphas for B
        :param betas: (K, K) matrix of prior betas for B
        """
        super(RBMSBM, self).__init__()
        self.N = N
        self.K = K
        self.M = M
        self.alphas = alphas
        self.betas = betas

        # Initialize the weights and biases for RBM
        self.W = xavier_init(self.M, self.K)
        self.b = -2 * np.ones((self.M, 1))
        self.c = -2 * np.ones((self.K, 1))

        # Initialize the parameters for posterior on block matrix
        self.alphas_post = alphas
        self.betas_post = betas

        # Initialize gradients
        self.grad_b = np.zeros(self.b.shape)
        self.grad_c = np.zeros(self.c.shape)
        self.grad_W = np.zeros(self.W.shape)

        # Initialize posterior for class membership
        self.q = np.ones((self.N, self.K))
        self.q = self.q / np.expand_dims(self.q.sum(axis=1), axis=1).repeat(repeats=self.K, axis=1)

    def variational_e_step(self, edges, A, Y, indices=None, lamda=0.5):
        """
        :param edges: List of edges in the network
        :param A: (N, N) binary adjacency matrix
        :param Y: (N, M) binary node feature matrix
        :param lamda: Regularization parameter
        :param indices: List of indices of nodes that are to be updated
        """
        if indices is None:
            indices = range(self.N)

        # Update the posterior on B
        q_prod_edges = np.zeros((self.K, self.K))
        for i, j in edges:
            q_prod_edges += np.matmul(self.q[i, :].reshape((-1, 1)), self.q[j, :].reshape((1, -1)))

        q_sum = self.q.sum(axis=0).reshape((-1, 1))
        residue = np.matmul(self.q.T, self.q)
        q_prod_all = np.matmul(q_sum, q_sum.T) - residue

        self.alphas_post = q_prod_edges + self.alphas
        self.betas_post = q_prod_all - q_prod_edges + self.betas

        # Update the posterior on community memberships
        digamma_alphas = digamma(self.alphas_post)
        digamma_betas = digamma(self.betas_post)
        digamma_sum = digamma(self.alphas_post + self.betas_post)

        q_alpha_prod = np.matmul(self.q, digamma_alphas)
        q_beta_prod = np.matmul(self.q, digamma_betas)
        q_sum_prod = np.matmul(self.q, digamma_sum)
        q_alpha_prod_t = np.matmul(self.q, digamma_alphas.T)
        q_beta_prod_t = np.matmul(self.q, digamma_betas.T)
        q_sum_prod_t = np.matmul(self.q, digamma_sum.T)

        def h(x, l=0.5):
            return (x <= 0.5) * (x**l * 2**(l-1)) + (x > 0.5) * (1 - 2**(l-1) * (1 - x)**l)

        for idx in indices:
            a_rep = np.asarray(A[idx, :].todense().reshape((-1, 1)).repeat(self.K, axis=1))
            a_rep_comp = 1 - a_rep

            temp1 = (a_rep * (q_alpha_prod - q_sum_prod)).sum(axis=0)
            temp2 = (a_rep_comp * (q_beta_prod - q_sum_prod)).sum(axis=0) - q_beta_prod_t[idx, :] + q_sum_prod_t[idx, :]

            a_rep = np.asarray(A[:, idx].todense().reshape((-1, 1)).repeat(self.K, axis=1))
            a_rep_comp = 1 - a_rep

            temp3 = (a_rep * (q_alpha_prod_t - q_sum_prod_t)).sum(axis=0)
            temp4 = (a_rep_comp * (q_beta_prod_t - q_sum_prod_t)).sum(axis=0) - q_beta_prod_t[idx, :] + \
                    q_sum_prod_t[idx, :]

            self.q[idx, :] = temp1 + temp2 + temp3 + temp4 + np.matmul(Y[idx, :].todense(), self.W) + self.c[:, 0]
            self.q[idx, :] = softmax(self.q[idx, :], axis=0)
            self.q[idx, :] = h(self.q[idx, :], lamda)
            self.q[idx, :] = softmax(self.q[idx, :], axis=0, take_exp=False)

    def variational_m_step(self, Y, lr=1e-2, momentum=0.0):
        """
        :param Y: (N, M) observed binary feature matrix
        :param lr: Learning rate for parameter updates
        :param momentum: Momentum term for SGD update
        """
        # Compute p(y_m = 1|z_k = 1) and p(z_k = 1)
        p_yz = sigmoid(self.W + np.tile(self.b, reps=(1, self.K)))
        p_z = np.exp(np.sum(np.log(1 + np.exp(self.W + np.tile(self.b, reps=(1, self.K)))), axis=0)) * \
              np.exp(self.c).reshape((-1))
        p_z = softmax(p_z, take_exp=False).reshape((-1, 1))

        # Compute expectations
        yz_mean = p_yz * np.tile(p_z.T, reps=(self.M, 1))
        y_mean = np.sum(yz_mean, axis=1).reshape((-1, 1))
        z_mean = p_z

        # Compute gradients for RBM parameters
        grad_b = -self.N * y_mean + Y.sum(axis=0).reshape((-1, 1))
        grad_c = -self.N * z_mean + self.q.sum(axis=0).reshape((-1, 1))
        grad_W = -self.N * yz_mean + np.matmul(Y.T.todense(), self.q)

        # Update the RBM parameters
        self.grad_b = momentum * self.grad_b + (1 - momentum) * grad_b / self.N
        self.b += lr * self.grad_b
        self.grad_c = momentum * self.grad_c + (1 - momentum) * grad_c / self.N
        self.c += lr * self.grad_c
        self.grad_W = momentum * self.grad_W + (1 - momentum) * grad_W / self.N
        self.W += lr * self.grad_W

        self.W = self.W.clip(-5, 5)
        self.b = self.b.clip(-5, 5)
        self.c = self.c.clip(-5, 5)
