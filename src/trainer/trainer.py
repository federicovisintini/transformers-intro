import typing
from datetime import datetime

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            loss_fn: nn.Module | typing.Callable,
            optimizer: Optimizer,
            lr_scheduler: LRScheduler
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.log_batch_timeout = 10  # log after N batches

    def _train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, batch in tqdm(enumerate(self.train_dataloader)):
            # Extract label from batch
            labels = batch['labels']

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(batch)

            # swap token_ids with num_tokens dimensions, to match torch loss syntax
            outputs = torch.swapaxes(outputs, -2, -1)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Adjust learning rate of optimizer
            self.lr_scheduler.step()

            # Gather data and report
            running_loss += loss.item()

            if i % self.log_batch_timeout == self.log_batch_timeout - 1:
                last_loss = running_loss / self.log_batch_timeout  # loss per batch
                learning_rate = self.lr_scheduler.get_last_lr()[0]
                print(f'  batch {i + 1} loss: {last_loss:.6f} | lr: {learning_rate:.2e}')
                tb_x = epoch_index * len(self.train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Learning Rate', learning_rate, tb_x)
                running_loss = 0.0

        return last_loss

    def _validate_one_epoch(self):
        running_vloss = 0.0
        for i, vdata in enumerate(self.validation_dataloader):
            vinputs, vlabels = vdata
            voutputs = self.model(vinputs)
            vloss = self.loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        return avg_vloss

    def fit(self, num_epochs=5):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(f'../runs/wmt14_trainer_{timestamp}')

        best_vloss = 1_000_000.

        for epoch in range(num_epochs):
            print(f'EPOCH {epoch + 1}:')

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self._train_one_epoch(epoch, writer)

            # We don't need gradients on to do reporting
            self.model.train(False)
            avg_vloss = self._validate_one_epoch()

            print(f'LOSS train {avg_loss} valid {avg_vloss}')

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch + 1)
            writer.flush()

            # Track the best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f'model_{timestamp}_{epoch}'
                torch.save(self.model.state_dict(), model_path)

    def validate(self):
        raise NotImplemented
