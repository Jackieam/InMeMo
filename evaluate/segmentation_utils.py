import numpy as np

from evaluate.mae_utils import WHITE, YELLOW, PURPLE, BLACK


def calculate_metric(args, target, ours, fg_color=WHITE, bg_color=BLACK):
    # Crop the right area:
    arr = args.arr
    if arr == 'a1' or arr == 'a8':
        target = target[113:, 113:]
        ours = ours[113:, 113:]
    elif arr == 'a2' or arr == 'a7':
        target = target[113:, :111]
        ours = ours[113:, :111]
    elif arr == 'a3' or arr == 'a6':
        target = target[:111, :111]
        ours = ours[:111, :111]
    elif arr == 'a4' or arr == 'a5':
        target = target[:111, 113:]
        ours = ours[:111, 113:]
    else:
        raise ValueError("the arrangement is not in list!")

    return _calc_metric(ours, target, fg_color, bg_color)


def _calc_metric(ours, target, fg_color=WHITE, bg_color=BLACK):
    fg_color = np.array(fg_color)
    # Calculate accuracy:
    accuracy = np.sum(np.float32((target == ours).all(axis=2))) / (ours.shape[0] * ours.shape[1])
    seg_orig = ((target - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    seg_our = ((ours - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    color_blind_seg_our = (ours - np.array([[bg_color]]) != 0).any(axis=2)
    iou = np.sum(np.float32(seg_orig & seg_our)) / np.sum(np.float32(seg_orig | seg_our))
    color_blind_iou = np.sum(np.float32(seg_orig & color_blind_seg_our)) / np.sum(
        np.float32(seg_orig | color_blind_seg_our))
    return {'iou': iou, 'color_blind_iou': color_blind_iou, 'accuracy': accuracy}


def get_default_mask_1row_mask():
    mask = np.zeros((14,14))
    mask[:7] = 1
    mask[:, :7] = 1
    return mask