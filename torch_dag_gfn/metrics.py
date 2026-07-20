"""Posterior-evaluation metrics (ported from the original DAG-GFlowNet, which
adapts them from DiBS, Lorch et al. 2021, MIT License)."""
import numpy as np

from sklearn import metrics


def expected_shd(posterior, ground_truth):
    """Expected Structural Hamming Distance between the posterior samples and the
    ground-truth graph. `posterior` is (B, N, N), `ground_truth` is (N, N)."""
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))
    diff = np.minimum(diff, 1)  # ignore double edges
    shds = np.sum(diff, axis=(1, 2)) / 2
    return float(np.mean(shds))


def expected_edges(posterior):
    """Expected number of edges in graphs sampled from the posterior."""
    return float(np.mean(np.sum(posterior, axis=(1, 2))))


def threshold_metrics(posterior, ground_truth):
    """AUROC / PRC-AUC / average precision of the posterior edge marginals against
    the ground-truth adjacency."""
    p_edge = np.mean(posterior, axis=0).reshape(-1)
    gt = ground_truth.reshape(-1)

    fpr, tpr, _ = metrics.roc_curve(gt, p_edge)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt, p_edge)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt, p_edge)

    return {
        'roc_auc': float(roc_auc),
        'prc_auc': float(prc_auc),
        'ave_prec': float(ave_prec),
    }
