import torch
import torch.nn.functional as F


def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = torch.eq(torch.max(preds, 1)[1], labels)
    return torch.mean(correct_prediction.float())


def expected_rmse(logits, labels, class_values=None):
    """
    Computes the root mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth label
    :param class_values: rating values corresponding to each class.
    :return: rmse
    """

    probs = F.softmax(logits, 1)
    if class_values is None:
        #scores = tf.to_float(tf.range(start=0, limit=logits.get_shape()[1]) + 1)
        scores = torch.range(start=0, end=logits.size(1)).float() + 1.
        y = labels.float() + 1.  # assumes class values are 1, ..., num_classes
    else:
        scores = class_values
        y = torch.index_select(class_values, 0, labels)

    pred_y = torch.sum(probs * scores, 1)

    diff = torch.sub(y, pred_y)
    exp_rmse = torch.pow(diff, 2)

    return torch.sqrt(torch.mean(exp_rmse))


def rmse(logits, labels, class_values=None):
    """
    Computes the mean square error with the predictions
    computed as average predictions. Note that without the average
    this cannot be used as a loss function as it would not be differentiable.
    :param logits: predicted logits
    :param labels: ground truth labels for the ratings, 1-D array containing 0-num_classes-1 ratings
    :param class_values: rating values corresponding to each class.
    :return: mse
    """

    if class_values is None:
        y = tf.to_float(labels) + 1.  # assumes class values are 1, ..., num_classes
    else:
        y = tf.gather(class_values, labels)

    pred_y = logits

    diff = tf.subtract(y, pred_y)
    mse = tf.square(diff)
    mse = tf.cast(mse, dtype=tf.float32)

    return tf.sqrt(tf.reduce_mean(mse))


def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """

    loss = F.cross_entropy(input=outputs, target=labels, reduction='mean')
    return loss
