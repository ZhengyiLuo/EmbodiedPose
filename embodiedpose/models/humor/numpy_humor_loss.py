import math

import numpy as np

from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    (The code is borrowed from https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367)

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1,
                                leaf_size=1,
                                algorithm='kd_tree',
                                metric=metric).fit(x)
        chamfer_dist = x_nn.kneighbors(y)[0]
        # chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1,
                                leaf_size=1,
                                algorithm='kd_tree',
                                metric=metric).fit(y)
        chamfer_dist = y_nn.kneighbors(x)[0]
        # chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1,
                                leaf_size=1,
                                algorithm='kd_tree',
                                metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1,
                                leaf_size=1,
                                algorithm='kd_tree',
                                metric=metric).fit(y)
        chamfer_dist = y_nn.kneighbors(x)[0]
        # chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError(
            "Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'"
        )

    return chamfer_dist


def robust_std(res):
    '''
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (N, 1)

    Returns:
    - std : 1
    '''
    med = np.median(res, axis=0)
    abs_dev = np.abs(res - med)
    MAD = np.median(abs_dev, axis=0)
    std = MAD / 0.67449
    return std


def bisquare_robust_weights(res, tune_const=4.6851):
    '''
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    '''
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
    #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2)**2
    w[outlier_mask] = 0.0

    return w


def apply_robust_weighting(res,
                           robust_loss_type='bisquare',
                           robust_tuning_const=4.6851):
    '''
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    '''
    robust_choices = ['none', 'bisquare']
    if robust_loss_type not in robust_choices:
        print('Not a valid robust loss: %s. Please use %s' %
              (robust_loss_type, str(robust_choices)))

    w = None
    detach_res = np.copy(res)
    if robust_loss_type == 'none':
        w = np.ones_like(detach_res)
    elif robust_loss_type == 'bisquare':
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w


def motion_prior_loss(z, pm, pv):
    log_prob = -np.log(np.sqrt(pv)) - math.log(math.sqrt(2 * math.pi)) - (
        (z - pm)**2 / (2 * pv))
    log_prob = -np.sum(log_prob, axis=0)
    return log_prob


def points3d_loss(points3d_obs, points3d_pred):
    # one-way chamfer
    N_obs, _ = points3d_obs.shape
    N_pred = points3d_pred.shape[0]
    points3d_obs = points3d_obs.reshape((-1, 3))
    points3d_pred = points3d_pred.reshape((-1, 3))

    obs2pred_sqr_dist = chamfer_distance(points3d_obs, points3d_pred)
    obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((N_obs))

    weighted_obs2pred_sqr_dist, w = apply_robust_weighting(
        np.sqrt(obs2pred_sqr_dist), robust_loss_type='bisquare')

    loss = np.mean(weighted_obs2pred_sqr_dist)
    return loss