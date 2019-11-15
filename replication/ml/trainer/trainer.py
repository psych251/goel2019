import math
import os
import pickle
from typing import List, Optional
import datetime as dt

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from replication.ml.network.network import TouchNet

eps = 1e-8

SAVE_FILE_NAME = "last.save.npy"


def append_to(array: Optional[np.array], new_array: torch.Tensor):
    new_array_np = new_array.cpu().detach().numpy()
    if len(new_array_np.shape) <= 1:
        new_array_np = np.array([new_array_np])
    if array is None:
        return new_array_np
    else:
        return np.concatenate((array, new_array_np), axis=0)


class TouchTrainer:
    writer: SummaryWriter
    output_dir: str
    writer_dir: str
    model: torch.nn.Module

    def __init__(self, model: TouchNet, train_loader, val_loader, test_loader, device, max_step=10000000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_step = max_step
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=1e-4)
        self.n_iter = 0
        self.min_avg_loss = math.inf
        self.device = device

    def save_to(self, last_checkpoint, file_dir):
        save_file_content = {
            "output_dir": self.output_dir,
            "writer_dir": self.writer_dir,
            "save": last_checkpoint,
        }
        with open(file_dir, "wb") as save_file:
            pickle.dump(save_file_content, save_file)

    def load_from(self, file_dir):
        with open(file_dir, "rb") as save_file:
            save_file_content = pickle.load(save_file)

        checkpoint_path = save_file_content["save"]
        print(f"checkpoint file: {checkpoint_path}")
        with open(checkpoint_path, "rb") as model_file:
            checkpoint = torch.load(model_file)
        self.n_iter = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.output_dir = save_file_content["output_dir"]
        self.writer_dir = save_file_content["writer_dir"]

    def to_device(self, data: List[torch.Tensor]):
        return [tensor.to(self.device) for tensor in data]

    def process_input(self, input_a: List[torch.Tensor], input_b: List[torch.Tensor]):
        input_a = self.to_device(input_a)
        input_b = self.to_device(input_b)
        output = self.model(input_a + input_b)
        output_a = output[0: len(input_a)]
        output_b = output[len(input_a): len(input_a) + len(input_b)]
        # reference = torch.zeros((len(input_a)), dtype=torch.long).to(self.device)
        assert len(input_a) == len(input_b)
        # noinspection PyArgumentList
        reference = torch.LongTensor([[0]] * len(input_a)).to(self.device)
        aligned_output = torch.stack((output_a, output_b), dim=1)
        loss = self.criterion(aligned_output, reference)
        # noinspection PyUnresolvedReferences
        correct_rate = (output_a[:] > output_b[:]).float().mean()
        # noinspection PyArgumentList
        return loss, correct_rate

    def train_iter(self, input_a: List[torch.Tensor], input_b: List[torch.Tensor]):
        self.model.train()
        loss, correct_rate = self.process_input(input_a, input_b)
        self.writer.add_scalar('train/loss', loss, self.n_iter)
        self.writer.add_scalar('train/rate', correct_rate, self.n_iter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def data_loader_generator(data_loader: DataLoader):
        data_loader_iterator = iter(data_loader)
        while True:
            try:
                data = next(data_loader_iterator)
            except StopIteration:
                data_loader_iterator = iter(data_loader)
                data = next(data_loader_iterator)
            yield data

    # noinspection PyShadowingBuiltins
    def train(self, resume_from: Optional[str]):
        torch.autograd.set_detect_anomaly(True)

        output_parent_dir = "./checkpoints"
        if resume_from is None:
            current_datetime = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            self.output_dir = os.path.join(output_parent_dir, current_datetime)
            self.writer_dir = os.path.join("./runs/", current_datetime)
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
        else:
            save_file_dir = os.path.join(output_parent_dir, resume_from)
            save_file_path = os.path.join(save_file_dir, SAVE_FILE_NAME)
            self.load_from(save_file_path)

        print(self.writer_dir)

        self.writer = SummaryWriter(log_dir=self.writer_dir)
        evaluation_done = False
        testing_done = False
        train_gen = self.data_loader_generator(self.train_loader)
        while self.n_iter < self.max_step or True:
            if self.n_iter % 100 == 0 and (not evaluation_done):
                self.eval()
                evaluation_done = True
            elif self.n_iter % 1000 == 0 and (not testing_done):
                print("testing...")
                self.eval(True, checkpoint=False)
                testing_done = True
            else:
                input_a, input_b = next(train_gen)
                self.train_iter(input_a, input_b)
                self.n_iter += 1
                evaluation_done = False
                testing_done = False

    def eval(self, test=False, checkpoint=True):
        self.model.eval()
        loss_array: Optional[List[float]] = None
        rate_array: Optional[List[float]] = None
        data_loader = self.test_loader if test else self.val_loader
        prefix = "test/" if test else "val/"
        for input_a, input_b in data_loader:
            loss, correct_rate = self.process_input(input_a, input_b)
            loss_array = append_to(loss_array, loss)
            rate_array = append_to(rate_array, correct_rate)

        avg_loss = np.average(loss_array)
        self.writer.add_scalar(prefix + 'avg_loss', avg_loss, self.n_iter)
        avg_rate = np.average(rate_array)
        self.writer.add_scalar(prefix + 'avg_rate', avg_rate, self.n_iter)

        if (self.n_iter % 1000 == 0 or avg_loss < self.min_avg_loss) and checkpoint:
            self.min_avg_loss = avg_loss
            filename = f"modelTrained_{self.n_iter}_{avg_loss}.pickle"
            print(f"Creating checkpoint: {filename}")
            filepath = os.path.join(self.output_dir, filename)
            torch.save({'epoch': self.n_iter,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': avg_loss}, filepath)
            self.save_to(filepath, os.path.join(self.output_dir, SAVE_FILE_NAME))
        return avg_loss

    def close(self, path):
        self.writer.export_scalars_to_json(path)
        self.writer.close()
