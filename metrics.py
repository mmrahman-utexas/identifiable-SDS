import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


# def reconstruction_mse_per_frame(output, target, **kwargs):

#     # return np.mean(np.sum((np.squeeze(output) - np.squeeze(target)) ** 2, axis=(2, 3)))
    
#     """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
#     output = output.reshape(output.shape[0], output.shape[1], output.shape[3], output.shape[4])
#     target = target.reshape(target.shape[0], target.shape[1], target.shape[3], target.shape[4])
#     mean = np.sum((output - target) ** 2, axis=(2, 3)).mean()
#     return mean

def reconstruction_mse_per_frame(output, target, **kwargs):
    """
    Computes the mean and standard deviation of the per-frame reconstruction MSE.
    Returns:
        mean (float): average sum-of-squares error per frame over the batch.
        std  (float):   standard deviation of sum-of-squares error per frame.
    """
    # reshape from (B, F, C, H, W) → (B, F, H, W)
    output = output.reshape(output.shape[0], output.shape[1], output.shape[3], output.shape[4])
    target = target.reshape(target.shape[0], target.shape[1], target.shape[3], target.shape[4])

    # sum-of-squares per sample per frame → shape (B, F)
    errors = np.sum((output - target) ** 2, axis=(2, 3))

    mean = float(errors.mean())
    std  = float(errors.std())
    return mean, std


def mean_corr_coef(x, y, method='pearson'):
    """
    Compute the mean correlation coefficient between the predicted and ground truth states.
    """
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score, cc