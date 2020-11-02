import numpy as np


def get_scores(mask, valve, mask_gt, valve_gt):

    # NOTE: mask here means the box (ROI)!

    if len(mask.shape) == 3:
        mask = np.squeeze(mask[..., 0])  # predicted mask should be the same everywhere

    dices = []
    ious = []
    for v in valve_gt:
        frame_idx = int(list(v.keys())[0]) - 1  # label numbering starts at 1, python indexing at 0
        target = list(v.values())[0]
        pred = np.squeeze(valve[..., frame_idx])

        iou = _get_iou(pred, target)
        iou_reverse = _get_iou(target, pred)

        assert iou == iou_reverse

        dice = _get_dice(target, pred)

        ious.append(iou)
        dices.append(dice)

    window_acc = _get_window_acc(mask_gt, mask)
    window_iou = _get_iou(mask_gt, mask)

    return {'iou': np.mean(ious), 'dice': np.mean(dices), 'window_acc': window_acc, 'window_iou': window_iou,
            'dice_1': dices[0], 'dice_2': dices[1], 'dice_3': dices[2], 'iou_1': ious[0], 'iou_2':
                ious[1], 'iou_3': ious[2]}


def _get_iou(target, prediction):
    """
    Computes the dice coefficient of the window detection and valve according to formula (5.2)
    :param target:
    :param prediction:
    :return:
    """
    target = np.asarray(target).astype(np.bool)
    prediction = np.asarray(prediction).astype(np.bool)

    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def _get_dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def _get_window_acc(mask_gt, mask_pred):

    """
    Computes the window detection accuracy according to formula (5.1) Master's Thesis Jesse.
    :param mask_gt:
    :param mask_pred:
    :return:
    """

    mask_gt = mask_gt.astype(np.bool)
    mask_pred = mask_pred.astype(np.bool)

    intersection = np.logical_and(mask_gt, mask_pred)

    acc = np.sum(intersection) / np.sum(mask_pred)

    return acc
