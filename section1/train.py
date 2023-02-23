import abc
import enum

import torch
from torch import nn
from tqdm.auto import tqdm
import click

from unet import Unet
#from scaler import StaticScaler, DynamicScaler

from dataset import get_train_data


class ScalerUpdateStatus(enum.Enum):
    SUCCESS = "SUCCESS"
    FOUND_INF_OR_NAN = "FOUND_INF_OR_NAN"


class LossScaler(abc.ABC):
    def __init__(self, init_scale: float):
        self._loss_scale = init_scale
        self.steps_since_last_inf_nan = 0

    @staticmethod
    def _has_inf_or_nan(x: torch.Tensor) -> bool:
        return not torch.isfinite(x).all() or torch.isnan(x).any()

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self._loss_scale

    def step(self, optimizer: torch.optim.Optimizer) -> ScalerUpdateStatus:
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                assert param.grad.dtype == torch.float32, (
                    f"Some gradient is not in FP32.")
                if self._has_inf_or_nan(param.grad):
                    self.steps_since_last_inf_nan = 0
                    optimizer.zero_grad()

                    return ScalerUpdateStatus.FOUND_INF_OR_NAN

                param.grad /= self._loss_scale

        optimizer.step()
        optimizer.zero_grad()
        self.steps_since_last_inf_nan += 1

        return ScalerUpdateStatus.SUCCESS

    @abc.abstractmethod
    def update(self):
        """Update self._loss_scale."""


class StaticScaler(LossScaler):
    def __init__(self, init_scale: float):
        super().__init__(init_scale)

    def update(self):
        pass  # No updates here :).


class DynamicScaler(LossScaler):
    def __init__(self, init_scale=65536, grow_factor=2,
                 shrink_factor=0.5, consecutive_steps=10):
        super().__init__(init_scale)

        self.grow_factor = grow_factor
        self.shrink_factor = shrink_factor
        self.consecutive_steps = consecutive_steps

    def update(self):
        if self.steps_since_last_inf_nan == 0:
            self._loss_scale *= self.shrink_factor
        elif self.steps_since_last_inf_nan == self.consecutive_steps:
            self._loss_scale *= self.grow_factor


def train_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()

    #pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        print(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")
        #pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


@click.command()
@click.option('--grad-scaler',
              type=click.Choice(['static', 'dynamic'], case_sensitive=False),
              default='static'
              )
def train(grad_scaler):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scaler = StaticScaler() if grad_scaler == 'static' else DynamicScaler()

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, scaler, device=device)


if __name__ == '__main__':
    train()
