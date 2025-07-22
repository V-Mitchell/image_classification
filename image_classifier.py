import os
import shutil
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from models.classification_models import get_model
from utils.dict import remove_key

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class CIFAR10Dataset(Dataset):
    train_batch_files = [
        "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"
    ]
    test_batch_files = ["test_batch"]

    def __init__(self, dataset_path, test=False):
        super().__init__()
        images = []
        labels = []
        if test:
            batch_files = self.test_batch_files
        else:
            batch_files = self.train_batch_files

        for batch_file in batch_files:
            data = self._unpickle(os.path.join(dataset_path, batch_file))
            batch_images = torch.from_numpy(data["data".encode()]).to(dtype=torch.uint8,
                                                                      device=torch.device("cpu"))
            batch_labels = torch.tensor(data["labels".encode()],
                                        dtype=torch.long,
                                        device=torch.device("cpu"))
            batch_images = batch_images.view(-1, 3, 32, 32)
            images.append(batch_images)
            labels.append(batch_labels)
        self.images = torch.cat(images).float() / 255.0  # normalize image data
        self.labels = torch.cat(labels)

    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    def visualize(self):
        save_dir = "dataset"
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.__len__()):
            image, label = self.__getitem__(i)
            print(os.path.join(save_dir, str(i) + "_" + LABELS[label.data] + ".png"))
            save_image(
                image.squeeze(),
                os.path.join(save_dir,
                             str(i) + "_" + LABELS[label.data] + ".png"),
            )

    def collate_fn(self, batch):
        batch_images = []
        batch_labels = []
        for x in batch:
            image, label = x
            batch_images.append(image.unsqueeze(0))
            batch_labels.append(label.unsqueeze(0))
        batch_images = torch.cat(batch_images)
        batch_labels = torch.cat(batch_labels)
        return (batch_images, batch_labels)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return (self.images[index], self.labels[index])


def get_dataloader(cfg, test=False):
    dataset = CIFAR10Dataset(cfg["dataset_path"], test)
    return DataLoader(dataset,
                      cfg["batch_size"] if not test else 1,
                      cfg["shuffle"] if not test else False,
                      num_workers=cfg["num_workers"],
                      collate_fn=dataset.collate_fn,
                      pin_memory=cfg["pin_memory"],
                      drop_last=cfg["drop_last"],
                      persistent_workers=cfg["persistent_workers"])


OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "ASGD": torch.optim.ASGD
}
SCHEDULERS = {
    "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "MultiStepLR": torch.optim.lr_scheduler.MultiStepLR,
}


def get_optimizer(cfg, model_params):
    return OPTIMIZERS[cfg["type"]](model_params, **remove_key(cfg, ["type"]))


def get_scheduler(cfg, optimizer):
    return SCHEDULERS[cfg["type"]](optimizer, **remove_key(cfg, ["type"]))


class TrainingLogger():
    def __init__(self, base_log_path):
        self.log_path = os.path.join(base_log_path, datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))
        self.writer = SummaryWriter(self.log_path)
        self.log_dict_step = 0

        self.cfg_dir_path = os.path.join(self.log_path, "cfg")
        os.makedirs(self.cfg_dir_path)
        self.weights_dir_path = os.path.join(self.log_path, "weights")
        os.makedirs(self.weights_dir_path)
        logging.basicConfig(filename=os.path.join(self.log_path, "session.log"),
                            level=logging.INFO,
                            format="%(asctime)s/%(levelname)s/%(message)s")
        self.logger = logging.getLogger(__name__)

    def log_dict(self, data_dict):
        for k, v in data_dict.items():
            self.writer.add_scalar(k, v, self.log_dict_step)
        self.log_dict_step += 1

    def save_cfg(self, cfg_path):
        shutil.copy(cfg_path, os.path.join(self.cfg_dir_path, os.path.basename(cfg_path)))

    def save_checkpoint(self, epoch, model):
        torch.save(model.state_dict(),
                   os.path.join(self.weights_dir_path,
                                str(epoch) + "_checkpoint.pth"))

    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.weights_dir_path, "image_classifier.pth"))

    def log_message(self, msg_str):
        self.logger.info(msg_str)

    def get_log_path(self):
        return self.log_path


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def bce_loss(preds, labels):
    label_one_hot = F.one_hot(labels.long(), 10).float()
    loss = F.binary_cross_entropy_with_logits(preds, label_one_hot, reduction="mean")
    return loss


def train_epoch(epoch, model, dataloader, optimizer, device, logger):
    last_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} - Loss {last_loss}")
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        images, labels = data
        labels = labels.to(device=device)
        images = images.to(device=device)
        preds = model(images)
        loss = bce_loss(preds, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.log_dict({"loss": loss})
        last_loss = loss.item()
        pbar.set_description(f"Epoch {epoch} - Loss {last_loss}")


def validate(epoch, model, dataloader, device, logger):
    model.eval()
    confusion_mat = np.zeros((10, 10), dtype=np.int64)
    pbar = tqdm(dataloader, desc=f"Validation")
    for i, data in enumerate(pbar):
        images, labels = data
        labels = labels.to(device=device)
        images = images.to(device=device)
        preds = model(images)
        preds = F.sigmoid(preds)
        preds_argmax = torch.argmax(preds, -1)
        confusion_mat[labels.item()][preds_argmax.item()] += 1

    tp = confusion_mat.trace()
    total = confusion_mat.sum()
    accuracy = tp.astype(float) / total.astype(float)
    logger.log_message(f"\n{confusion_mat}")
    logger.log_message(f"Epoch {epoch} Accuracy: {accuracy}")

    model.train()


def train(cfg):
    if cfg["training"]["deterministic"]:
        seed = 0xffff_ffff_ffff_ffff
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")
    model = get_model(cfg["model"])
    model = model.to(device=device)
    model.apply(weights_init)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model Parameters: {params}".format(params=num_params))

    optimizer = get_optimizer(cfg["training"]["optimizer"], model.parameters())
    scheduler = None
    if "scheduler" in cfg["training"]:
        scheduler = get_scheduler(cfg["training"]["scheduler"], optimizer)

    train_dataloader = get_dataloader(cfg["dataloader"])
    test_dataloader = get_dataloader(cfg["dataloader"], True)
    logger = TrainingLogger(cfg["training"]["log_path"])
    logger.save_cfg(cfg["cfg_path"])

    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        train_epoch(epoch, model, train_dataloader, optimizer, device, logger)
        if cfg["training"][
                "validate_period"] > 0 and epoch % cfg["training"]["validate_period"] == 0:
            validate(epoch, model, test_dataloader, device, logger)
            logger.save_checkpoint(epoch, model)

        if scheduler is not None:
            scheduler.step()

    # final validation and model save
    validate(epoch, model, test_dataloader, device, logger)
    logger.save_model(model)


if __name__ == "__main__":
    import yaml
    import argparse
    parser = argparse.ArgumentParser(description="Image Classifier Engine")
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--test', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--visualize',
                        type=bool,
                        action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()
    with open(args.cfg) as stream:
        cfg = yaml.safe_load(stream)
    cfg["cfg_path"] = args.cfg
    # make training engine into a class
    train(cfg)
