import os
import json
import random
import datetime
import numpy as np

from tqdm import tqdm
from sklearn.metrics import average_precision_score

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from loaders import get_loaders, adjust_data
from model2 import TDL
from eval import Evaluation
from losses import CombinedLoss


def get_filename(cfg):
    return "_".join(
        map(
            str,
            [
                cfg["dataset_name"],
                cfg["av_hubert_name"],
                cfg["max_length"],
                cfg["model"]["input_dimension"],
                cfg["model"]["d_model"],
                cfg["model"]["win_size"],
                f'e{cfg["model"]["encoder_nlayers"]["retain"]}' + f'{cfg["model"]["encoder_nlayers"]["downsample"]}',
                "-".join(cfg["model"]["reconstruction_modality"]),
                f'r{cfg["model"]["reconstruction_nlayers"]["pre"]}' + f'{cfg["model"]["reconstruction_nlayers"]["downsample"]}' + f'{cfg["model"]["reconstruction_nlayers"]["upsample"]}' + f'{cfg["model"]["reconstruction_nlayers"]["post"]}',
                f'r{cfg["model"]["reconstruction_error_encoder_nlayers"]}',
                "-".join(cfg["criterion"]),
            ],
        )
    )


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ["PYTHONHASHSEED"] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class Logger:
    def __init__(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.path = os.path.join(folder, f"{filename}.json")

    def create(self):
        with open(self.path, "w") as hundle:
            json.dump({}, hundle, indent=2)

    def update(self, key, content):
        with open(self.path, "r") as hundle:
            json_file = json.load(hundle)

        if isinstance(key, str):
            json_file[key] = content

        with open(self.path, "w") as hundle:
            json.dump(json_file, hundle, indent=2)


class Experiment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]
        self.seed = cfg["seed"]
        self.epochs = cfg["epochs"]
        self.dataset_name = cfg["dataset_name"]
        self.av_hubert_name = cfg["av_hubert_name"]
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.workers = cfg["workers"]
        self.input_dimension = cfg["model"]["input_dimension"]
        self.d_model = cfg["model"]["d_model"]
        self.win_size = cfg["model"]["win_size"]
        self.encoder_nlayers = cfg["model"]["encoder_nlayers"]
        self.reconstruction_modality = cfg["model"]["reconstruction_modality"]
        self.reconstruction_nlayers = cfg["model"]["reconstruction_nlayers"]
        self.reconstruction_error_encoder_nlayers = cfg["model"]["reconstruction_error_encoder_nlayers"]
        self.criterion_composition = cfg["criterion"]
        self.lr = cfg["learning_rate"]
        self.factor = [1] * self.encoder_nlayers["retain"] + [2 ** (i + 1) for i in range(self.encoder_nlayers["downsample"])]

        self.ckpt_dir = cfg["save_dir"]
        self.filename = get_filename(cfg)
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{self.filename}.pth")
        self.results = []

    def compute_metric(self, labels, predictions):
        return sum(
            [
                average_precision_score(
                    labels[:, :: self.factor[i], 0].cpu().numpy(),
                    torch.sigmoid(prediction_[:, :, 0]).detach().cpu().numpy(),
                    average="micro",
                )
                for i, prediction_ in enumerate(predictions)
            ]
        ) / len(predictions)

    def adjust_metrics(self, loss, metric_name, metric, validation):
        l = {"loss": validation["loss"]}
        m = {f"{x}@{y}": validation[x][y] for x in ["ap", "ar"] for y in validation[x]}
        validation = {**l, **m}
        pf_keys = ["vloss", "vap@0.5"]
        t = {"loss": loss, metric_name: 100 * metric}
        v = {f"v{x}": (99 * int(x != "loss") + 1) * validation[x] for x in validation}
        metrics = {**t, **v}
        postfix = {x: v[x] for x in pf_keys}
        postfix = {**t, **postfix}
        return metrics, postfix

    def train_one_epoch(self, epoch):
        start_epoch = datetime.datetime.now()
        self.model.train()
        tot_loss, tot_metric = 0, 0
        metric_name = "ap"
        with tqdm(unit="batch", total=len(self.loaders["train"]), dynamic_ncols=True, disable=False) as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            for iteration, data in enumerate(self.loaders["train"]):
                data_adjusted = adjust_data(data, self.device)
                video_features = data_adjusted["video_features"]
                audio_features = data_adjusted["audio_features"]
                p, z = self.model(video_features, audio_features)
                # [[64, 512, 3], [64, 512, 3], [64, 256, 3], [64, 128, 3]], [64, 512]
                self.optimizer.zero_grad()
                loss_ = self.criterion(p, data_adjusted["labels"], z)
                tot_loss += loss_.item()
                loss_.backward()
                self.optimizer.step()
                tot_metric += self.compute_metric(data_adjusted["labels"], p)
                tepoch.set_postfix({"loss": tot_loss / (iteration + 1), metric_name: 100.0 * tot_metric / (iteration + 1)})
                tepoch.update(1)

            e = Evaluation(
                model=self.model,
                loader=self.loaders["val"],
                criterion=self.criterion,
                device=self.device,
                factor=self.factor,
                max_length=self.max_length,
                dataset_name=self.dataset_name,
                task='tfl'
            )
            e.compute_metrics()
            metrics, postfix = self.adjust_metrics(
                loss=tot_loss / len(self.loaders["train"]),
                metric_name=metric_name,
                metric=tot_metric / len(self.loaders["train"]),
                validation=e.metrics,
            )
            duration = datetime.datetime.now() - start_epoch
            metrics["epoch"] = epoch + 1
            metrics["duration"] = str(duration)
            current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
            self.scheduler.step(current_score)
            tepoch.set_postfix(postfix)
            tepoch.close()
            return metrics, duration

    def training_process(self):
        best_score = 0
        best_epoch = 0
        metrics_train_val = []
        for epoch in range(self.epochs):
            metrics, duration = self.train_one_epoch(epoch)
            metrics_train_val.append(metrics)
            print(f"[{datetime.datetime.now()}: Epoch {epoch + 1}/{self.epochs} {str(duration)}]", metrics)
            self.logger.update("results", [{"training": metrics_train_val}])
            current_score = sum([metrics[x] for x in metrics if "vap" in x or "var" in x])
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"[{datetime.datetime.now()}] Saved checkpoint at {self.ckpt_path}")
            elif epoch - best_epoch >= 7:
                print(f"[{datetime.datetime.now()}] Early stopped training at epoch {epoch + 1}")
                break

        self.model.load_state_dict(torch.load(self.ckpt_path))
        e = Evaluation(
            model=self.model,
            loader=self.loaders["test"],
            criterion=self.criterion,
            device=self.device,
            factor=self.factor,
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            task='tfl'
        )
        e.compute_metrics()
        l = {"tloss": e.metrics["loss"]}
        m = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
        metrics_test = {**l, **m}
        print(json.dumps(metrics_test, indent=2))
        return metrics_train_val, metrics_test

    def run(self):
        print(f"[{datetime.datetime.now()}] Seed {self.seed} Experiment starts")
        self.logger = Logger(folder=self.ckpt_dir, filename=self.filename)
        print(self.logger.path)
        self.logger.create()
        start_time = datetime.datetime.now()
        seed_everything(self.seed)
        self.loaders = get_loaders(
            dataset_name=self.dataset_name,
            av_hubert_name=self.av_hubert_name,
            max_length=self.max_length,
            batch_size=self.batch_size,
            workers=self.workers,
            splits=None
        )
        self.model = TDL(
            input_dimension=self.input_dimension,
            max_length=self.max_length,
            d_model=self.d_model,
            win_size=self.win_size,
            encoder_nlayers=self.encoder_nlayers,
            reconstruction_modality=self.reconstruction_modality,
            reconstruction_nlayers=self.reconstruction_nlayers,
            reconstruction_error_encoder_nlayers=self.reconstruction_error_encoder_nlayers
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", patience=7)
        self.criterion = CombinedLoss(self.criterion_composition, self.factor)
        training, test = self.training_process()
        end_time = datetime.datetime.now()
        duration = str(end_time - start_time)
        self.logger.update("results", {"seed": self.seed, "training": training, "test": test, "duration": duration})
        print(f"duration: {duration}")
        print(f"[{datetime.datetime.now()}] Experiment ends")
