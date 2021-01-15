import numpy as np
import torch
from losses import dice_round


def dice_round_fn(predicted, ground_truth, score_threshold=0.5, area_threshold=0):
    """
    predicted, ground_truth: torch tensors
    """
    mask = predicted > score_threshold
    #     mask[mask.sum(dim=(1,2,3)) < area_threshold, :,:,:] = torch.zeros_like(mask[0])
    if mask.sum() < area_threshold:
        mask = torch.zeros_like(mask)
    #     print(1 - dice_round(mask, ground_truth).item())
    return 1 - dice_round(mask, ground_truth).item()


#
#     return  1 - dice_round(mask, ground_truth).item()


def search_thresholds(eval_list, thr_list, area_list):
    best_score = 0
    best_thr = -1
    best_area = -1

    for thr in thr_list:
        for area in area_list:
            score_list = []
            for probas, labels in eval_list:
                score = dice_round_fn(probas, labels, thr, area)
                score_list.append(score)
            final_score = np.mean(score_list)
            if final_score > best_score:
                best_score = final_score
                best_thr = thr
                best_area = area
    return best_thr, best_area, best_score
