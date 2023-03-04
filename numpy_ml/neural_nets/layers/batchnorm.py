import numpy as np

def bn_forward(x,gamma, beta, eps=1e-5):
    sample_mean = np.mean(x,axis=0)
    sample_var = np.var(x, axis=0)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    y = gamma * x_hat + beta
    cache = (x, x_hat, gamma, beta, sample_mean,sample_var, eps)
    return y, cache


def bn_backward(dy, cache):
    x, x_hat, gamma, beta, sample_mean,sample_var, eps = cache
    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(dy * x_hat, axis=0)


    dx_hat = dy * gamma
    dvar = np.sum(dx_hat * x_hat, axis=0) * (-1/2) * (1/eps+sample_var) * (1/len(dy)) * 2 * (x-sample_mean)
    dmean = np.sum(dx_hat,axis=0) / (-1) * np.sqrt(sample_var+eps) + dvar * (1/m) * (-2) * (sample_mean)
    dx_hat_xi = dx_hat * (1/np.sqrt(sample_var+eps))

    dx = dvar+dmean+dx_hat_xi

    return dx, dgamma, dbeta