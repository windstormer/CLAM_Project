import numpy as np

def compute_dice(gt, gen):
    # print(gen.dtype, gt.dtype)
    gt = gt.astype(np.uint8)
    gen = gen.astype(np.uint8)
    # print(gt.shape, gen.shape)
    inse = np.logical_and(gt, gen).sum()
    dice = (2. * inse + 1e-5) / (np.sum(gt) + np.sum(gen) + 1e-5)
    return dice

def compute_mIOU(gt, gen):
    gt = gt.astype(np.uint8)
    gen = gen.astype(np.uint8)
    intersection = np.logical_and(gt, gen)
    # print(intersection)
    union = np.logical_or(gt, gen)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score