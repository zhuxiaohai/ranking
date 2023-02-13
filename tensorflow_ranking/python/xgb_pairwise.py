import numpy as np
import math
from scipy import special
from sklearn.metrics import ndcg_score, roc_auc_score
import tensorflow as tf
from tensorflow_ranking.python import losses_impl


@tf.function
def pairwise_rank(scores_loop, labels_loop, scores, labels):
    finalg = []
    finalh = []
    for i in range(len(scores_loop)):
        p = scores_loop[i]
        y = labels_loop[i]
        lambda_weight = losses_impl.DCGLambdaWeight()
        loss_fn = losses_impl.PairwiseHingeLossArbitraryMargin(margin=0, name=None, lambda_weight=lambda_weight)
        with tf.GradientTape(persistent=True) as h:
            h.watch(p)
            h.watch(scores)
            with tf.GradientTape(persistent=True) as g:
                g.watch(p)
                g.watch(scores)
                logits = tf.nn.sigmoid(scores)
                logits_p = tf.nn.sigmoid(p)
                loss_final = loss_fn.compute_1vM(y, logits_p, labels, logits,
                                                 weights=None,
                                                 reduction=tf.compat.v1.losses.Reduction.MEAN)
            dz_dx = g.gradient(loss_final, scores)
            dz_dp = g.gradient(loss_final, p)
            dz_dx_sum = tf.reduce_sum(dz_dx) + dz_dp
        dz_dx2 = h.gradient(dz_dx_sum, scores)
        dz_dp2 = h.gradient(dz_dp, p)
        dz_dx2_sum = tf.reduce_sum(dz_dx2) + dz_dp2
        finalg.append(dz_dx_sum)
        finalh.append(dz_dx2_sum)
    return tf.stack(finalg), tf.stack(finalh)


def bce_margin_loss(predt, dtrain, w_margin=3, w_bce=1):
    scores = special.logit(predt)
    y_true = dtrain.get_label()
    y_01 = np.array([0 if item == 0 else 1 for item in y_true])
    labels = y_true
    batchsize = 100
    batch_num = math.ceil(len(labels) / batchsize)
    g_list = []
    h_list = []
    for i in range(batch_num):
        scores_loop = tf.constant(scores[i*batchsize:batchsize*(i+1)])
        labels_loop = tf.constant(labels[i*batchsize:batchsize*(i+1)])
        g, h = pairwise_rank(scores_loop, labels_loop, tf.constant(scores), tf.constant(labels))
        g_list.append(g)
        h_list.append(h)
    finalg = w_margin * tf.concat(g_list, axis=0).numpy() + w_bce * (predt - y_01)
    finalh = w_margin * tf.concat(h_list, axis=0).numpy() + w_bce * predt * (1 - predt)
    return finalg, finalh


def bce_margin_loss_legacy(predt, dtrain, w_margin=3, w_bce=1, margin=0, mask=[]):
    y_true = dtrain.get_label()
    y_01 = np.array([0 if item == 0 else 1 for item in y_true])
    predt_label = [item for item in zip(predt, y_true)]
    gra = []
    hra = []
    for p, y in predt_label:
        if y in mask:
            gra.append(0)
            hra.append(0)
            continue
        count = 0
        sumg = 0
        sumh = 0
        count += (y < y_true).sum()
        candidate = predt[(y < y_true) & ((p + margin) > predt)]
        if candidate.shape[0] > 0:
            sumg += (p * (1 - p) - candidate * (1 - candidate)).sum()
            sumh += ((p - 3 * p * p + 2 * p * p * p) - (
                      candidate - 3 * candidate * candidate + 2 * candidate * candidate * candidate)).sum()
        count += (y > y_true).sum()
        candidate = predt[(y > y_true) & (p < (predt + margin))]
        if candidate.shape[0] > 0:
            sumg += (-p * (1 - p) + candidate * (1 - candidate)).sum()
            sumh += (-(p - 3 * p * p + 2 * p * p * p) + (
                      candidate - 3 * candidate * candidate + 2 * candidate * candidate * candidate)).sum()
        if count > 0:
            sumg = sumg / count
            sumh = sumh / count
            gra.append(sumg)
            hra.append(sumh)
        else:
            gra.append(0)
            hra.append(0)
    g = w_margin * np.asarray(gra) + w_bce * (predt - y_01)
    h = w_margin * np.asarray(hra) + w_bce * predt * (1 - predt)
    return g, h


def margin_loss_metric(predt, dtrain, margin=0.001, mask=[]):
    y_true = dtrain.get_label()
    predt_label = [item for item in zip(predt, y_true)]
    losses = []
    for p, y in predt_label:
        if y in mask:
            losses.append(0)
            continue
        loss = 0
        count = 0
        count += (y < y_true).sum()
        candidate = predt[(y < y_true) & ((p + margin) > predt)]
        loss += (p - candidate).sum()
        count += (y > y_true).sum()
        candidate = predt[(y > y_true) & (p < (predt + margin))]
        loss += (-p + candidate).sum()
        if count > 0:
            loss = loss / count
        else:
            loss = 0
        losses.append(loss)
    return 'margin_score', np.array(losses).mean()


def bce_loss_metric(predt, dtrain):
    y_true = dtrain.get_label()
    y_01 = np.array([0 if item == 0 else 1 for item in y_true])
    losses = -(y_01 * (np.log(predt)) + (1 - y_01) * (np.log(1 - predt)))
    return 'bce_score', np.array(losses).mean()


def ndcg_metric(predt, dtrain, mask=[]):
    y_true = dtrain.get_label()
    indices = [False if i in mask else True for i in y_true]
    score = ndcg_score(np.array(y_true[indices]).reshape(1, -1), predt[indices].reshape(1, -1))
    return 'ndcg_score', score


def auc_metric(predt, dtrain):
    y_true = dtrain.get_label()
    y_01 = np.array([0 if item == 0 else 1 for item in y_true])
    score = roc_auc_score(y_01, predt)
    return 'auc_score', score