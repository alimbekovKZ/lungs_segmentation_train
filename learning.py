import torch
from torch.nn.utils import clip_grad_norm_
import pandas as pd
from tqdm import tqdm
import heapq
from pathlib import Path

torch.multiprocessing.set_sharing_strategy("file_system")


class Learning:
    def __init__(
        self,
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
        logger,
    ):
        self.logger = logger

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.n_epoches = n_epoches
        self.scheduler = scheduler
        self.freeze_model = freeze_model
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.early_stopping = early_stopping

        self.calculation_name = calculation_name
        self.best_checkpoint_path = Path(
            best_checkpoint_folder, f"{self.calculation_name}.pth"
        )
        self.checkpoints_history_folder = Path(checkpoints_history_folder)
        self.checkpoints_topk = checkpoints_topk
        self.score_heap = []
        self.summary_file = Path(self.checkpoints_history_folder, "summary.csv")

        self.best_score = 0
        self.best_epoch = -1

    def train_epoch(self, model, loader):
        tqdm_loader = tqdm(loader)
        #         model.train()
        current_loss_mean = 0

        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):

            loss, predicted = self.batch_train(model, imgs, labels, batch_idx)

            # just slide average
            current_loss_mean = (current_loss_mean * batch_idx + loss) / (batch_idx + 1)

            tqdm_loader.set_description(
                "loss: {:.4} lr:{:.6}".format(
                    current_loss_mean, self.optimizer.param_groups[0]["lr"]
                )
            )
        return current_loss_mean

    def batch_train(self, model, batch_imgs, batch_labels, batch_idx):
        batch_imgs, batch_labels = (
            batch_imgs.permute(0, 3, 1, 2).float().to(self.device),
            batch_labels.float().permute(0, 3, 1, 2).to(self.device),
        )
        predicted = model(batch_imgs)
        loss = self.loss_fn(predicted, batch_labels)

        loss.backward()
        if batch_idx % self.grad_accum == self.grad_accum - 1:
            clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), predicted

    def valid_epoch(self, model, loader, local_metric_fn):
        tqdm_loader = tqdm(loader)
        eval_list = []
        current_score_mean = 0
        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            imgs = imgs.permute(0, 3, 1, 2).float()
            labels = labels.permute(0, 3, 1, 2).float()
            with torch.no_grad():
                predicted = self.batch_valid(model, imgs)
                eval_list.append((predicted, labels))

                score = local_metric_fn(predicted, labels)
                current_score_mean = (current_score_mean * batch_idx + score) / (
                    batch_idx + 1
                )

                tqdm_loader.set_description(f"score: {current_score_mean:.5}")

        return eval_list, current_score_mean

    def batch_valid(self, model, batch_imgs):
        batch_imgs = batch_imgs.to(self.device)
        predicted = model(batch_imgs)
        predicted = torch.sigmoid(predicted)
        return predicted.cpu()

    def process_summary(self, eval_list, epoch, global_metric_fn, mean_loss):
        self.logger.info(f"{epoch} epoch: \t start searching thresholds....")

        selected_thr, selected_area, selected_score = global_metric_fn(eval_list)

        epoch_summary = pd.DataFrame(
            data=[[epoch, selected_thr, selected_area, selected_score, mean_loss]],
            columns=[
                "epoch",
                "best_score_thr",
                "best_area_thr",
                "best_metric",
                "valid_loss",
            ],
        )

        self.logger.info(f"{epoch} epoch: \t Best score threshold: {selected_thr:.3}")
        self.logger.info(f"{epoch} epoch: \t Best area threshold: {selected_area}")
        self.logger.info(f"{epoch} epoch: \t Calculated score: {selected_score:.5}")

        if epoch == 0:
            epoch_summary.to_csv(self.summary_file, index=False)
        else:
            summary = pd.read_csv(self.summary_file)
            summary = summary.append(epoch_summary)
            summary.to_csv(self.summary_file, index=False)

        return selected_score

    def post_processing(self, score, epoch, model):
        checkpoints_history_path = Path(
            self.checkpoints_history_folder,
            f"{self.calculation_name}_epoch{epoch}.pth",
        )
        torch.save(model.state_dict(), checkpoints_history_path)
        heapq.heappush(self.score_heap, (score, checkpoints_history_path))
        if len(self.score_heap) > self.checkpoints_topk:
            _, removing_checkpoint_path = heapq.heappop(self.score_heap)
            removing_checkpoint_path.unlink()
            self.logger.info(f"Removed checkpoint is {removing_checkpoint_path}")
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.best_checkpoint_path)
            self.logger.info(f"best model: {epoch} epoch - {score:.5}")

        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def run_train(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        local_metric_fn,
        global_metric_fn,
    ):
        model.to(self.device)
        for epoch in range(self.n_epoches):
            if not self.freeze_model:
                self.logger.info(f"{epoch} epoch: \t start training....")
                model.train()
                train_loss_mean = self.train_epoch(model, train_dataloader)
                self.logger.info(
                    "{} epoch: \t Calculated train loss: {:.5}".format(
                        epoch, train_loss_mean
                    )
                )

            self.logger.info(f"{epoch} epoch: \t start validation....")
            model.eval()
            eval_list, mean_loss = self.valid_epoch(
                model, valid_dataloader, local_metric_fn
            )

            selected_score = self.process_summary(
                eval_list, epoch, global_metric_fn, mean_loss
            )

            self.post_processing(selected_score, epoch, model)

            if epoch - self.best_epoch > self.early_stopping:
                self.logger.info("EARLY STOPPING")
                break

        return self.best_score, self.best_epoch
