import argparse

from torch.utils.data import DataLoader
import torch

import importlib
import functools
import os
from pathlib import Path

from dataset import LungDataset
from learning import Learning
from utils.helpers import load_yaml, init_seed, init_logger
from evaluation import dice_round_fn, search_thresholds
import transforms as T


def argparser():
    parser = argparse.ArgumentParser(description="Lung Segmentation pipeline")
    parser.add_argument("train_cfg", type=str, help="train config path")
    return parser.parse_args()


def init_eval_fns(train_config):
    score_threshold = train_config["EVALUATION"]["SCORE_THRESHOLD"]
    area_threshold = train_config["EVALUATION"]["AREA_THRESHOLD"]
    thr_search_list = train_config["EVALUATION"]["THRESHOLD_SEARCH_LIST"]
    area_search_list = train_config["EVALUATION"]["AREA_SEARCH_LIST"]
    local_metric_fn = functools.partial(
        dice_round_fn, score_threshold=score_threshold, area_threshold=area_threshold
    )

    global_metric_fn = functools.partial(
        search_thresholds, thr_list=thr_search_list, area_list=area_search_list
    )
    return local_metric_fn, global_metric_fn


def train(
    train_config,
    experiment_folder,
    pipeline_name,
    log_dir,
    train_dataloader,
    valid_dataloader,
    local_metric_fn,
    global_metric_fn,
):

    fold_logger = init_logger(log_dir, "train.log")

    best_checkpoint_folder = Path(
        experiment_folder, train_config["CHECKPOINTS"]["BEST_FOLDER"]
    )
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        experiment_folder,
        train_config["CHECKPOINTS"]["FULL_FOLDER"],
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config["CHECKPOINTS"]["TOPK"]

    calculation_name = f"{pipeline_name}"

    device = train_config["DEVICE"]

    module = importlib.import_module(train_config["MODEL"]["PY"])
    model_class = getattr(module, train_config["MODEL"]["CLASS"])
    model = model_class(**train_config["MODEL"]["ARGS"])
    if len(train_config["DEVICE_LIST"]) > 1:
        model = torch.nn.DataParallel(model)

    pretrained_model_config = train_config["MODEL"].get("PRETRAINED", False)

    if pretrained_model_config:
        loaded_pipeline_name = pretrained_model_config["PIPELINE_NAME"]
        pretrained_model_path = Path(
            pretrained_model_config["PIPELINE_PATH"],
            pretrained_model_config["CHECKPOINTS_FOLDER"],
            f"{loaded_pipeline_name}.pth",
        )
        fold_logger.info(f"load model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

    module = importlib.import_module(train_config["CRITERION"]["PY"])
    loss_class = getattr(module, train_config["CRITERION"]["CLASS"])
    loss_fn = loss_class(**train_config["CRITERION"]["ARGS"])

    optimizer_class = getattr(torch.optim, train_config["OPTIMIZER"]["CLASS"])
    optimizer = optimizer_class(model.parameters(), **train_config["OPTIMIZER"]["ARGS"])
    scheduler_class = getattr(
        torch.optim.lr_scheduler, train_config["SCHEDULER"]["CLASS"]
    )
    scheduler = scheduler_class(optimizer, **train_config["SCHEDULER"]["ARGS"])

    n_epoches = train_config["EPOCHES"]
    grad_clip = train_config["GRADIENT_CLIPPING"]
    grad_accum = train_config["GRADIENT_ACCUMULATION_STEPS"]
    early_stopping = train_config["EARLY_STOPPING"]

    freeze_model = train_config["MODEL"]["FREEZE"]

    Learning(
        optimizer,
        loss_fn,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        calculation_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        fold_logger,
    ).run_train(
        model, train_dataloader, valid_dataloader, local_metric_fn, global_metric_fn
    )


def main():
    args = argparser()
    config_folder = Path(args.train_cfg.strip("/"))
    experiment_folder = config_folder.parents[0]

    train_config = load_yaml(config_folder)

    log_dir = Path(experiment_folder, train_config["LOGGER_DIR"])
    log_dir.mkdir(exist_ok=True, parents=True)

    main_logger = init_logger(log_dir, "train_main.log")

    seed = train_config["SEED"]
    init_seed(seed)
    main_logger.info(train_config)

    if "DEVICE_LIST" in train_config:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            map(str, train_config["DEVICE_LIST"])
        )

    pipeline_name = train_config["PIPELINE_NAME"]

    img_size = train_config["IMAGE_SIZE"]
    train_transform, valid_transform = T.transformA(img_size)

    data_folder = train_config["DATA_DIRECTORY"]
    mask_folder = train_config["MASK_DIRECTORY"]
    num_workers = train_config["WORKERS"]
    batch_size = train_config["BATCH_SIZE"]
    local_metric_fn, global_metric_fn = init_eval_fns(train_config)
    train_index = train_config["TRAIN_INDEX"]
    masks = os.listdir(f"{mask_folder}ml/")
    train_masks = masks[:train_index]
    test_masks = masks[train_index:]
    main_logger.info("Start training ...")

    train_dataset = LungDataset(
        train_masks,
        data_folder=data_folder,
        mask_folder=mask_folder,
        mode="train",
        transforms=train_transform,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    valid_dataset = LungDataset(
        test_masks,
        data_folder=data_folder,
        mask_folder=mask_folder,
        mode="test",
        transforms=valid_transform,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset, batch_size=1, num_workers=num_workers, shuffle=False
    )

    train(
        train_config,
        experiment_folder,
        pipeline_name,
        log_dir,
        train_dataloader,
        valid_dataloader,
        local_metric_fn,
        global_metric_fn,
    )


if __name__ == "__main__":
    main()
