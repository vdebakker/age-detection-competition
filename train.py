import numpy as np
import pandas as pd

import torch
from torch import optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm, trange

from dataloading import get_dataloader
from model import get_model, save_model
from utils import cycle, ema_avg
from loss import loss_func


train_acc, train_loss, train_steps = 0, 0, 0
valid_acc, valid_loss, valid_steps = 0, 0, 0


def get_scheduler(optimizer, max_lr=2e-4, steps_per_epoch=500, epochs=100, pct_start=.01):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
        epochs=epochs, pct_start=pct_start)


def get_optimizer(params, lr=2e-4, weight_decay=1e-5):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_step(model, train_dl, optimizer, scheduler):
    global train_acc, train_loss, train_steps

    model.train()

    train_steps += 1
    bunch = next(train_dl)
    imgs, ages = bunch

    imgs = imgs.to('cuda')
    ages = ages.to('cuda')

    for p in model.parameters(): p.grad = None

    outs = model(imgs)

    loss = loss_func(outs, ages)
    loss.backward()

    optimizer.step()
    scheduler.step()

    train_loss = loss.item() * (1 - gamma) + gamma * train_loss
    train_acc = (torch.argmax(outs.detach(), dim=-1) == torch.argmax(ages, dim=-1)).float().mean().cpu() * (1 - gamma) + gamma * train_acc


def validate(model, valid_dl, validation_steps=500):
    global valid_loss, valid_acc, valid_steps

    model.eval()
    with torch.no_grad():
        for _ in range(validation_steps):
            bunch = next(valid_dl)

            valid_steps += 1
            imgs, ages = bunch

            imgs = imgs.to('cuda')
            ages = ages.to('cuda')

            outs = model(imgs)
            loss = loss_func(outs, ages)

            valid_loss = loss.item() * (1 - gamma) + gamma * valid_loss
            valid_acc = (torch.argmax(outs.detach(), dim=-1) == torch.argmax(ages, dim=-1)).float().mean().cpu() * (1 - gamma) + gamma * valid_acc


if __name__ == '__main__':
    cudnn.benchmark = True

    torch.multiprocessing.set_start_method('spawn')

    classes = ['YOUNG', 'MIDDLE', 'OLD']

    model_path = 'resnet18.pt'

    batch_size = 32
    epochs = 5
    max_lr = .0002
    smoothing = 0
    augment_prob = .9
    augment_prob2 = .4
    pct_start = .02
    weight_decay = 	.00001
    dropout_prob = 0.

    steps_per_epoch = 620
    gamma = .99

    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train_df['ID'] = 'data/train/' + train_df['ID']
    test_df['ID'] = 'data/test/' + test_df['ID']

    train_dl = get_dataloader(train_df, batch_size=batch_size, include_labels=True, augment_prob=augment_prob, augment_prob2=augment_prob2, num_workers=8)
    test_dl = get_dataloader(test_df, batch_size=batch_size, include_labels=False, augment_prob=0, num_workers=8)

    train_dl = cycle(train_dl)

    model = get_model(dropout_prob)

    avg_model = optim.swa_utils.AveragedModel(model, avg_fn=ema_avg, device='cuda')

    model.cuda()

    optimizer = get_optimizer(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = get_scheduler(optimizer, max_lr=max_lr,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs, pct_start=pct_start)

    for epoch in range(epochs):
        for _ in trange(steps_per_epoch):
            train_one_step(model, train_dl, optimizer, scheduler)
            avg_model.update_parameters(model)

        print(f'----------   Epoch {epoch}   ----------')
        print(f'Train loss: {train_loss / (1 - gamma ** train_steps):.3f}')
        print(f'Train acc:  {train_acc / (1 - gamma ** train_steps):.3f}')

    optim.swa_utils.update_bn(test_dl, avg_model, device='cuda')

    valid_acc, valid_loss, valid_steps = 0, 0, 0

    validate(avg_model.module, train_dl, 500)

    print(f'Avg Train loss: {valid_loss / (1 - gamma ** valid_steps):.3f}')
    print(f'Avg Train acc:  {valid_acc / (1 - gamma ** valid_steps):.3f}')

    save_model(avg_model.module, model_path)