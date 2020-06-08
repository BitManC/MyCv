import torch
import tqdm
from torch_lr_finder import LRFinder
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, dl_train, criterion, optimizer,
                 dl_valid=None, dl_test=None, device="cuda",
                 save_dir=None, metrics=None, on_train_val_epoch_finished_callback=None):
        self.model = model
        self.dl_train = dl_train
        self.num_train_samples = len(dl_train.dataset)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dl_valid = dl_valid
        self.num_validation_samples = len(dl_valid.dataset)

        self.dl_test = dl_test
        self.device = device
        self.save_dir = save_dir
        self.metrics = metrics
        self.recorder = MetricRecorder(self.metrics)
        self.latest_lr_finder_result = None
        self.on_train_val_epoch_finished_callback = on_train_val_epoch_finished_callback

    def train(self, num_epochs, max_lr=1e-4, hide_progress=False):
        self.model.to(self.device)
        self.model.train()
        print("start to train ")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                        steps_per_epoch=len(self.dl_train),
                                                        epochs=num_epochs)

        self.recorder.reset()

        for epoch in range(num_epochs, desc="Epochs", disable=hide_progress):
            self._train_one_epoch(epoch, scheduler, hide_progress)

            if self.dl_valid is not None:
                self._validate(epoch, hide_progress)

            if self.on_train_val_epoch_finished_callback is not None:
                self.on_train_val_epoch_finished_callback(epoch)

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _train_one_epoch(self, epoch, scheduler, hide_progress):
        self.model.train()

        for img_batch, mask_batch in tqdm(self.dl_train, desc=f"Epoch {epoch}", leave=False,
                                          disable=hide_progress, position=0):
            img_batch, mask_batch = img_batch.to(self.device), mask_batch.to(self.device)

            self.optimizer.zero_grad()

            model_output = self.model(img_batch)

            loss = self.criterion(model_output, mask_batch)

            with torch.no_grad():
                self.recorder.update_record_on_batch_end(epoch, loss.item(), mask_batch,
                                                         model_output.squeeze(),
                                                         img_batch.size(0),
                                                         self.num_train_samples)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        with torch.no_grad():
            self.recorder.finalize_record_on_epoch_end()

        tqdm.write(self.recorder.get_latest_epoch_message(training=True))

    def _validate(self, epoch, hide_progress):
        assert self.dl_valid is not None
        self.model.eval()

        with torch.no_grad():
            for img_batch, mask_batch in tqdm(self.dl_valid, desc="Validating",
                                              disable=hide_progress, leave=False, position=0):
                img_batch, mask_batch = img_batch.to(self.device), mask_batch.to(self.device)
                model_output = self.model(img_batch)
                loss = self.criterion(model_output, mask_batch)

                self.recorder.update_record_on_batch_end(epoch, loss.item(), mask_batch,
                                                         model_output.squeeze(),
                                                         img_batch.size(0),
                                                         self.num_validation_samples,
                                                         training=False)
            self.recorder.finalize_record_on_epoch_end(training=False)
            tqdm.write(self.recorder.get_latest_epoch_message(training=False))

    def lr_range_test(self, val_loss=False):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)

        val_loader = self.dl_valid if val_loss else None

        lr_finder.range_test(self.dl_train, val_loader=val_loader, end_lr=100,
                             num_iter=100, step_mode="exp")

        lr_finder.plot()
        lr_finder.reset()
        self.latest_lr_finder_result = lr_finder

    def predict(self, image_input, threshold=0.5, image_preprocessing_cb=None):
        self.model.eval()

        if image_preprocessing_cb is not None:
            image_input = image_preprocessing_cb(image_input)

        with torch.no_grad():
            image_input = image_input.to(self.device)
            model_out = self.model(image_input)

        return model_out


class MetricRecorder:
    def __init__(self, metrics):
        self.data = {"training": [], "validation": []}
        self.metrics = metrics

    def reset(self):
        self.data = {"training": [], "validation": []}

        for metric in self.metrics:
            metric.reset()

    def update_record_on_batch_end(self, epoch, loss, actual, prediction,
                                   n_batch_samples, n_total_samples,
                                   threshold=0.5, training=True):
        records = self.data["training" if training else "validation"]

        if epoch >= len(records):
            record = defaultdict(float)
            record["epoch"] = epoch
            records.append(record)
        else:
            record = records[epoch]

        bs_ratio = n_batch_samples / n_total_samples

        record["loss"] += loss * bs_ratio

        if self.metrics is not None:
            pred_proba = (torch.sigmoid(prediction) > threshold).type(actual.dtype)
            for metric in self.metrics:
                metric.on_batch_end(actual, pred_proba, bs_ratio)

    def finalize_record_on_epoch_end(self, training=True):
        record = self.data["training" if training else "validation"][-1]

        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict.update(metric.on_epoch_end())

        record.update(metrics_dict)

    def get_records_dataframe(self, training=True):
        return pd.DataFrame(self.data["training" if training else "validation"]).set_index("epoch")

    def get_latest_epoch_message(self, training=True):
        record_type = "Train" if training else "Valid"
        record = self.data["training" if training else "validation"][-1]

        message = "Epoch {0} - {1}: loss={2:.6f}".format(record["epoch"], record_type, record["loss"])
        metric_message = ', '.join(f"{key}={val:.4f}" for key, val in record.items()
                                   if key is not "epoch" and key is not "loss")
        if metric_message:
            message += ", " + metric_message
        return message

    def show_train_val_metric_curve(self, metric="loss", figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)

        train_pd = self.get_records_dataframe()
        val_pd = self.get_records_dataframe(training=False)

        train_pd.plot(y=metric, ax=ax, label=f"train-{metric}")
        val_pd.plot(y=metric, ax=ax, label=f"val-{metric}")
        plt.show()


class MetricCallback:

    def on_batch_end(self, actual, prediction, bs_ratio):
        pass

    def on_epoch_end(self):
        pass

    def reset(self):
        pass


class AccuracyMetric(MetricCallback):
    name = "acc"

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = torch.tensor(0.0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.acc += (prediction == actual).float().mean() * bs_ratio

    def on_epoch_end(self):
        result = {self.name: self.acc.item()}
        self.reset()

        return result


class PrecisionRecallF1Metric(MetricCallback):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0)
        self.tn = torch.tensor(0)
        self.fp = torch.tensor(0)
        self.fn = torch.tensor(0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.tp += (actual * prediction).sum()
        self.tn += ((1 - actual) * (1 - prediction)).sum()
        self.fp += ((1 - actual) * prediction).sum()
        self.fn += (actual * (1 - prediction)).sum()

    def on_epoch_end(self):
        sum_tp_fp = self.tp + self.fp
        precision = self.tp.float() / sum_tp_fp.float() if sum_tp_fp is not 0 else 0

        sum_tp_fn = self.tp + self.fn
        recall = self.tp.float() / sum_tp_fn.float() if sum_tp_fn is not 0 else 0

        sum_precison_recall = precision + recall
        f1 = 2.0 * precision * recall / sum_precison_recall.float() if sum_precison_recall is not 0 else 0

        self.reset()

        return {"prec": precision.item(), "recall": recall.item(), "f1": f1.item()}


class IOUMetric(MetricCallback):
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.intersection = torch.tensor(0)
        self.union = torch.tensor(0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.intersection += (actual & prediction).sum()
        self.union += (actual | prediction).sum()

    def on_epoch_end(self):
        result = (self.intersection.float() + self.smooth) / (self.union.float() + self.smooth)
        self.reset()
        return {"iou": result.item()}


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.3, bce_pos_weight = 1, dice_smooth=1):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_pos_weight = bce_pos_weight
        self.dice_smooth = dice_smooth

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        prob_inputs = torch.sigmoid(inputs)
        targets = targets.type(torch.float32)

        intersection = (prob_inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.dice_smooth) / (prob_inputs.sum() + targets.sum() + self.dice_smooth)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean',
                                                 pos_weight=torch.tensor(self.bce_pos_weight))
        result = self.bce_weight * bce + (1 - self.bce_weight) * dice_loss

        return result
