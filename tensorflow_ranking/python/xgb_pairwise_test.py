import numpy as np
from scipy import special
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import xgboost as xgb

import tensorflow as tf
from tensorflow_ranking.python import xgb_pairwise, losses_impl


def pairwise_rank(scores_loop, labels_loop, scores, labels, weights, mask, lambda_weights=None):
    losses = []
    for i in range(len(scores_loop)):
        p = scores_loop[i]
        y = labels_loop[i]
        if weights:
            w = weights[i]
            w = tf.cast(w, tf.float32)
        else:
            w = None
        if lambda_weights:
            lambda_weights = losses_impl.DCGLambdaWeight()
        loss_fn = losses_impl.PairwiseHingeLossArbitraryMargin(margin=0, name=None, lambda_weight=lambda_weights)
        logits = tf.nn.sigmoid(scores)
        logits_p = tf.nn.sigmoid(p)
        loss_final = loss_fn.compute_1vM(y, logits_p, labels, logits,
                                         weights=w,
                                         reduction=tf.compat.v1.losses.Reduction.MEAN,
                                         mask=mask)
        if weights:
            losses.append(loss_final*w)
        else:
            losses.append(loss_final)
    return tf.stack(losses)


def bce_margin_loss(scores, y_true, weights=None, mask=None, lambda_weights=True):
    labels = y_true
    batchsize = 2
    batch_num = math.ceil(len(y_true) / batchsize)
    l_list = []
    for i in range(batch_num):
        scores_loop = tf.constant(scores[i*batchsize:batchsize*(i+1)])
        labels_loop = tf.constant(labels[i*batchsize:batchsize*(i+1)])
        if weights:
            weights_loop = tf.constant(weights[i*batchsize:batchsize*(i+1)])
        else:
            weights_loop = None
        loss = pairwise_rank(scores_loop, labels_loop, tf.constant(scores), tf.constant(labels),
                             weights_loop, mask, lambda_weights)
        l_list.append(loss)
    return tf.concat(l_list, axis=0).numpy()


class XGBPairwiseHingeLossTest(tf.test.TestCase):

    def test_pairwise_hinge_loss(self):
        logits = np.array([0.4, 0.1, 0.6])
        labels = np.array([-1.0, -2.0, -3.0])
        losses = bce_margin_loss(logits, labels)
        hingeloss = lambda x: max(0, -x)
        scores = special.expit(logits)
        expected = [(hingeloss(scores[0] - scores[1]) + hingeloss(scores[0] - scores[2]))/2,
                    (hingeloss(scores[0] - scores[1]) + hingeloss(scores[1] - scores[2]))/2,
                    (hingeloss(scores[0] - scores[2]) + hingeloss(scores[1] - scores[2]))/2]
        self.assertAllClose(losses, expected)
        # a = bce_margin_loss(special.expit(np.array([0.4, 0.1, 0.6])).astype(np.float64), np.array([-1.0, -2.0, -3.0]))

    def test_pairwise_hinge_loss_per_list_weights(self):
        logits = np.array([0.4, 0.1, 0.6])
        labels = np.array([-1.0, -2.0, -3.0])
        weights = np.array([1.0, 2.0, 3.0])
        losses = bce_margin_loss(logits, labels, weights)
        hingeloss = lambda x: max(0, -x)
        scores = special.expit(logits)
        expected = [weights[0]*(hingeloss(scores[0] - scores[1]) + hingeloss(scores[0] - scores[2]))/2,
                    weights[1]*(hingeloss(scores[0] - scores[1]) + hingeloss(scores[1] - scores[2]))/2,
                    weights[2]*(hingeloss(scores[0] - scores[2]) + hingeloss(scores[1] - scores[2]))/2]
        self.assertAllClose(losses, expected)

    def test_pairwise_hinge_loss_mask(self):
        logits = np.array([0.4, 0.1, 0.6])
        labels = np.array([-1.0, -2.0, -3.0])
        mask = [True, False, True]
        losses = bce_margin_loss(logits, labels, mask=mask)
        hingeloss = lambda x: max(0, -x)
        scores = special.expit(logits)
        expected = [(0 + hingeloss(scores[0] - scores[2]))/1,
                    (hingeloss(scores[0] - scores[1]) + hingeloss(scores[1] - scores[2]))/2,
                    (hingeloss(scores[0] - scores[2]) + 0)/1]
        self.assertAllClose(losses, expected)

    def test_pairwise_hinge_loss_lambda_weights(self):
        logits = np.array([0.4, 0.1, 0.6])
        rank = np.array([2, 3, 1])
        labels = np.array([-1.0, -2.0, -3.0])
        losses = bce_margin_loss(logits, labels, lambda_weights=True)
        hingeloss = lambda x: max(0, -x)
        scores = special.expit(logits)
        weights01 = abs(labels[0] - labels[1])*abs(
            1/abs(rank[0]-rank[1])-1/(abs(rank[0]-rank[1])+1))*len(labels)
        weights02 = abs(labels[0] - labels[2]) * abs(
            1 / abs(rank[0] - rank[2]) - 1 / (abs(rank[0] - rank[2]) + 1)) * len(labels)
        weights12 = abs(labels[1] - labels[2]) * abs(
            1 / abs(rank[1] - rank[2]) - 1 / (abs(rank[1] - rank[2]) + 1)) * len(labels)
        expected = [(weights01 * hingeloss(scores[0] - scores[1]) +
                     weights02 * hingeloss(scores[0] - scores[2])) / (weights01 + weights02),
                    (weights01 * hingeloss(scores[0] - scores[1]) +
                     weights12 * hingeloss(scores[1] - scores[2])) / (weights01 + weights12),
                    (weights02 * hingeloss(scores[0] - scores[2]) +
                     weights12 * hingeloss(scores[1] - scores[2])) / (weights02 + weights12)]
        self.assertAllClose(losses, expected)

    def test_pairwise_hinge_loss_xgb(self):
        X_clf, y_clf = make_classification(n_samples=6000, n_features=20, n_classes=4,
                                           n_informative=4, n_redundant=6, random_state=0)

        X_clf_train, X_clf_valid, y_clf_train, y_clf_valid = train_test_split(
            X_clf, y_clf, test_size=0.3, shuffle=False)
        train_dmatrix = xgb.DMatrix(X_clf_train, y_clf_train)
        test_dmatrix = xgb.DMatrix(X_clf_valid,  y_clf_valid)

        params = {'max_depth': 2, 'eta': 0.01, 'eval_metric':['ndcg']}
        afsxc = xgb.train(params, train_dmatrix,
                          evals=[(test_dmatrix, 'test')],
                          obj=xgb_pairwise.bce_margin_loss,
                          feval=xgb_pairwise.auc_metric,
                          num_boost_round=20)