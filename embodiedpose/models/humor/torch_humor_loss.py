import math
import torch



def motion_prior_loss(z, pm, pv):
    log_prob = -torch.log(torch.sqrt(pv)) - math.log(math.sqrt(
        2 * math.pi)) - ((z - pm)**2 / (2 * pv))
    log_prob = -torch.sum(log_prob, dim=-1)

    return torch.mean(log_prob)


def chamfer_loss(point_cloud_obs, point_cloud_pred):
    from pytorch3d.ops import knn_points
    # loss_chamfer =
    # implement chamfer loss from scratch
    device = point_cloud_obs.device
    batch, src_n_points, src_n_dims = point_cloud_obs.shape
    _, tgt_n_points, tgt_n_dims = point_cloud_pred.shape

    src_nn = knn_points(
        point_cloud_obs,
        point_cloud_pred,
        lengths1=torch.tensor([src_n_points] * batch).to(device),
        lengths2=torch.tensor([tgt_n_points] * batch).to(device),
        K=1,
    )

    tgt_nn = knn_points(
        point_cloud_pred,
        point_cloud_obs,
        lengths1=torch.tensor([tgt_n_points] * batch).to(device),
        lengths2=torch.tensor([src_n_points] * batch).to(device),
        K=1,
    )

    return src_nn.dists, tgt_nn.dists


def points3d_loss(points3d_obs, points3d_pred, mean = True):
    # one-way chamfer
    B, T, N_obs, _ = points3d_obs.size()
    N_pred = points3d_pred.size(2)
    points3d_obs = points3d_obs.reshape((B * T, -1, 3))
    points3d_pred = points3d_pred.reshape((B * T, -1, 3))

    obs2pred_sqr_dist, pred2obs_sqr_dist = chamfer_loss(
        points3d_obs, points3d_pred)

    obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((B, T * N_obs))
    pred2obs_sqr_dist = pred2obs_sqr_dist.reshape((B, T * N_pred))
    weighted_obs2pred_sqr_dist, w = apply_robust_weighting(
        obs2pred_sqr_dist.sqrt())
    if mean:
        loss = weighted_obs2pred_sqr_dist.mean()
    else:
        loss = weighted_obs2pred_sqr_dist.reshape(B, T, -1, 1).mean(dim=2)
    loss = 0.5 * loss
    return loss


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
    detach_res = res.clone().detach(
    )  # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == 'none':
        w = torch.ones_like(detach_res)
    elif robust_loss_type == 'bisquare':
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w


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


def robust_std(res):
    ''' 
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    '''
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B, 1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    ​
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    ​
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv +
                          (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl