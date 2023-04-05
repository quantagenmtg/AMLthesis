import torch
from torch.linalg import inv, det


def cov_kernel(X1, X2, params):
    '''
    Exponential decay kernel plus additive noise used in Freeze-Thaw for learning curves

    :param X1: Training anchors input 1
    :param X2: Training anchors input 2
    :param params: [alpha, beta, sigma] parameters for the kernel
    :return: covariance matrix output at X1,X2
    '''
    alpha, beta, sigma = params[..., None, None]
    beta = torch.abs(beta)
    exponential1 = torch.sign(beta) * torch.pow(torch.abs(beta), alpha)
    sum = X1 + X2 + beta
    exponential2 = torch.sign(sum) * torch.pow(torch.abs(sum), alpha)
    exp_decay = exponential1 / exponential2
    noise = (sigma.clamp(0.001,1)**2) * (X1 == X2)
    return exp_decay + noise


def log_marginal_likelihood(params, y, X):
    '''
    Log marginal likelihood for optimization of hyperparameters for GP

    :param params: arameters for the covariance kernel and mean function
    :param y: actual outputs
    :param X: training anchors
    :return: log marginal likelihood
    '''
    # set mean
    mean = params[0]

    # make nans 0
    y = y - mean[..., None,:]
    nan = torch.isnan(y)
    nanT = nan.transpose(-1, -2)
    nan_ind = torch.where(nanT)
    y = torch.where(nan, torch.tensor(0.0), y)
    yT = y.transpose(-1, -2)

    # get covariance matrix and set rows/columns where nan to 0 as these are technically not datapoints
    cov = cov_kernel(X, X[..., None], params[1:])
    cov[nanT, :] = 0
    cov.transpose(-1, -2)[nanT, :] = 0
    cov[nan_ind[0], nan_ind[1], nan_ind[2], nan_ind[3], nan_ind[
        3]] = 1  # The trick is to set the "fake" datapoints diagonal to 1 and rest of its row and column to 1,
    # this way it is like it's not there for the determinant and inverse
    inv_cov = inv(cov)

    # calculate the matrix multiplication
    matmul = torch.einsum('...i,...ij,...j', yT, inv_cov, yT)

    # we also need the determinant of the covariance matrix
    # det_cov = det(cov)

    # and the number of actual datapoints
    # n = len(X) - nanT.sum(axis=-1)

    res = -1 / 2 * matmul #- 1 / 2 * torch.log(det_cov) - n / 2 * torch.log(torch.tensor(2 * torch.pi))
    return res
