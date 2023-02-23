import abc
import enum

import torch
from torch import nn
from tqdm.auto import tqdm
import click

from unet import Unet
from scaler import StaticScaler, DynamicScaler

from dataset import get_train_data


def train_epoch(train_loader, model, criterion, optimizer, scaler, device):
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
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

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


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
