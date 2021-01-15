import numpy as np
from glob import glob
import cv2
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import matplotlib.pyplot as plt


def argparser():
    parser = argparse.ArgumentParser(description="Prepare divison dataset")
    parser.add_argument(
        "-mask_path",
        default="../../Lung Segmentation/masks/",
        type=str,
        nargs="?",
        help="directory with masks",
    )
    parser.add_argument(
        "-out_path",
        default="../dataset",
        type=str,
        nargs="?",
        help="path for saving dataset",
    )
    parser.add_argument(
        "-n_threads", default=4, type=int, nargs="?", help="number of using threads"
    )
    return parser.parse_args()


def save_mask_file(f, out_path):
    img = plt.imread(f)
    img = img.astype("uint8")
    mask = cv2.connectedComponents(img)[1]
    uni = np.unique(mask, return_counts=True)
    indexes = [x for _, x in sorted(zip(uni[1], uni[0]))]
    indexes.pop(indexes.index(0))
    mask1 = mask.copy()
    mask[mask != indexes[-2]] = 0
    mask[mask == indexes[-2]] = 1
    mask1[mask1 != indexes[-1]] = 0
    mask1[mask1 == indexes[-1]] = 1
    mask = mask.astype("uint8")
    mask1 = mask1.astype("uint8")
    if np.where(mask != 0)[1].min() < np.where(mask1 != 0)[1].min():
        cv2.imwrite("{}/ml/{}.png".format(out_path, f[30:-4]), mask * 255)
        cv2.imwrite("{}/mr/{}.png".format(out_path, f[30:-4]), mask1 * 255)
    else:
        cv2.imwrite("{}/ml/{}.png".format(out_path, f[30:-4]), mask1 * 255)
        cv2.imwrite("{}/mr/{}.png".format(out_path, f[30:-4]), mask * 255)


def save_mask(mask_images_names, out_path="../dataset", n_threads=1):
    os.makedirs(out_path + "/ml", exist_ok=True)
    os.makedirs(out_path + "/mr", exist_ok=True)
    try:
        Parallel(n_jobs=n_threads, backend="threading")(
            delayed(save_mask_file)(f, out_path) for f in tqdm(mask_images_names)
        )
    except Exception as e:
        print(e)


def main():
    args = argparser()
    masks_fns = sorted(glob(f"{args.mask_path}/*.png"))
    out_path = args.out_path
    n_threads = args.n_threads
    save_mask(masks_fns, out_path, n_threads)


if __name__ == "__main__":
    main()
