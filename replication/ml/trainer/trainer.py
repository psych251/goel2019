import datetime as dt
import os
import pickle
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from replication.ml.data.data_splitter import DataSplitter
from replication.ml.network.network import TouchNet
from replication.ml.params import BATCH_MIN, USER_ITER, VAL_ITER, TEST_ITER

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
    # noinspection PyUnresolvedReferences
    optimizer: torch.optim.Optimizer
    save_files: Dict[str, Dict[int, str]]
    cached_scalar: Dict[str, Dict[int, Dict[str, float]]]

    def __init__(self, data_splitter: DataSplitter, device, max_step=10000000):
        self.data_splitter = data_splitter
        self.max_step = max_step
        self.criterion = nn.CrossEntropyLoss()
        self.n_iter = 0
        self.device = device
        self.user_names = self.data_splitter.user_names
        self.current_user_id = 0
        self.last_loss = None
        self.save_files = {user_name: {} for user_name in self.user_names}

        self.cached_scalar = {}

    @property
    def current_user_name(self):
        return self.user_names[self.current_user_id]

    def save_model(self, user_name: str):
        filename = f"modelTrained_{user_name}_{self.n_iter}_{self.last_loss}.pickle"
        print(f"Creating checkpoint: {filename}")
        filepath = os.path.join(self.output_dir, filename)
        torch.save({'epoch': self.n_iter,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.last_loss}, filepath)
        self.save_files[user_name][self.n_iter] = filepath
        self.save_to(os.path.join(self.output_dir, "save.pickle"))

    def load_model(self, user_name: str) -> bool:
        min_iter = -1
        checkpoint_path = None
        for n_iter, iter_checkpoint_path in self.save_files[user_name].items():
            if n_iter > min_iter:
                min_iter = n_iter
                checkpoint_path = iter_checkpoint_path
        if checkpoint_path is None:
            self.n_iter = 0
            # noinspection PyUnresolvedReferences
            self.model = TouchNet(7, 112).to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001, weight_decay=1e-4)
            return False
        print(f"checkpoint file: {checkpoint_path}")
        with open(checkpoint_path, "rb") as model_file:
            checkpoint = torch.load(model_file)
        self.n_iter = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return True

    def save_to(self, file_dir):
        save_file_content = {
            "current_user_id": self.current_user_id,
            "user_names": self.user_names,
            "save_files": self.save_files,
            "output_dir": self.output_dir,
            "writer_dir": self.writer_dir,
        }
        with open(file_dir, "wb") as save_file:
            pickle.dump(save_file_content, save_file)
#   self.save_to(filepath, os.path.join(self.output_dir, SAVE_FILE_NAME))

    def load_from(self, file_dir):
        with open(file_dir, "rb") as save_file:
            save_file_content = pickle.load(save_file)

        self.current_user_id = save_file_content["self.current_user_id"]
        self.user_names = save_file_content["user_names"]
        self.save_files = save_file_content["save_files"]
        self.output_dir = save_file_content["output_dir"]
        self.writer_dir = save_file_content["writer_dir"]

    def load_dir(self, file_dir):
        files = os.listdir(file_dir)
        self.current_user_id = -1
        self.model = TouchNet(7, 112).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0001, weight_decay=1e-4)
        print("model created")
        for file in files:
            name_segments = file.split('_')
            user_name = name_segments[1]
            iter = int(name_segments[2])
            self.save_files[user_name][iter] = os.path.join(file_dir, file)

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

    # def process_input(self, input_a: List[torch.Tensor], input_b: List[torch.Tensor]):
    #     input_a = self.to_device(input_a)
    #     input_b = self.to_device(input_b)
    #     output = self.model(input_a + input_b)
    #     output_a = output[0: len(input_a)]
    #     output_b = output[len(input_a): len(input_a) + len(input_b)]
    #     # reference = torch.zeros((len(input_a)), dtype=torch.long).to(self.device)
    #     assert len(input_a) == len(input_b)
    #     # noinspection PyArgumentList
    #     reference = torch.LongTensor([[1]] * len(input_a) + [[0]] * len(input_b)).to(self.device).float()
    #     loss = self.criterion(output, reference)
    #     # noinspection PyUnresolvedReferences
    #     correct_rate = ((output_a[:] > 0).float().mean() + (output_b[:] < 0).float().mean()) / 2
    #     # noinspection PyArgumentList
    #     return loss, correct_rate

    def train_iter(self, input_a: List[torch.Tensor], input_b: List[torch.Tensor]):
        self.model.train()
        loss, correct_rate = self.process_input(input_a, input_b)
        self.add_scalar('train/loss', loss.detach().cpu().numpy())
        self.add_scalar('train/rate', correct_rate.detach().cpu().numpy())
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
        evaluation_done = True
        testing_done = True
        train_gen = self.data_loader_generator(self.data_splitter.train_loader)

        while True:
            for self.current_user_id in range(len(self.user_names)):
            # for self.current_user_id in range(2):
                self.load_model(self.current_user_name)
                start_iter = self.n_iter
                print(f"User {self.current_user_name} iter {self.n_iter}-{self.n_iter + USER_ITER}")
                print(f"iter: ...", end='')
                while True:
                    print(f"\riter: {self.n_iter}", end='')
                    if self.n_iter - start_iter >= USER_ITER and evaluation_done and testing_done:
                        break
                    if self.n_iter % VAL_ITER == 0 and (not evaluation_done):
                        self.eval()
                        evaluation_done = True
                    elif self.n_iter % TEST_ITER == 0 and (not testing_done):
                        print("testing...")
                        self.eval(True, checkpoint=False)
                        testing_done = True
                    else:
                        input_data = next(train_gen)
                        filtered_input = [input_entry for (name, per_1, per_2), input_entry in input_data if
                                          name != self.current_user_name or
                                          (per_1 <= 0.1 and per_1 <= 0.1)]
                        if len(filtered_input) < BATCH_MIN:
                            if len(filtered_input) != 0:
                                print(f"Discarded {len(filtered_input)} train data")
                            continue
                        input_a, input_b = zip(*filtered_input)
                        self.train_iter(input_a, input_b)
                        self.n_iter += 1
                        evaluation_done = False
                        testing_done = False
                self.save_model(self.current_user_name)
            self.add_combined_scalar()

    def add_scalar(self, tag: str, value: float):
        if tag not in self.cached_scalar:
            self.cached_scalar[tag] = {}
        if self.n_iter not in self.cached_scalar[tag]:
            self.cached_scalar[tag][self.n_iter] = {}
        self.cached_scalar[tag][self.n_iter][self.current_user_name] = value
        self.writer.add_scalar(f"{tag}_{self.current_user_name}", value, self.n_iter)

    def add_combined_scalar(self):
        for tag, iterations in self.cached_scalar.items():
            for n_iter, values in iterations.items():
                avg_value = np.array(list(values.values())).mean()
                self.writer.add_scalar(tag, avg_value, n_iter)
        self.cached_scalar = {}

    def eval(self, test=False, checkpoint=True):
        self.model.eval()
        loss_array: Optional[List[float]] = None
        rate_array: Optional[List[float]] = None
        data_loader = self.data_splitter.test_loader[self.current_user_name] if test else self.data_splitter.val_loader
        prefix = "test/" if test else "val/"
        for input_data in data_loader:
            if test:
                filtered_input = [input_entry for (name, per_1, per_2), input_entry in input_data if per_1 > 0.1 and per_1 > 0.1]
            else:
                filtered_input = [input_entry for (name, per_1, per_2), input_entry in input_data if name != self.current_user_name]
            if len(filtered_input) < BATCH_MIN:
                if len(filtered_input) != 0:
                    print(f"Discarded {len(filtered_input)} eval/test data")
                continue
            input_a, input_b = zip(*filtered_input)
            loss, correct_rate = self.process_input(input_a, input_b)
            loss_array = append_to(loss_array, loss)
            rate_array = append_to(rate_array, correct_rate)

        avg_loss = np.average(loss_array)
        self.add_scalar(prefix + 'avg_loss', avg_loss)
        avg_rate = np.average(rate_array)
        self.add_scalar(prefix + 'avg_rate', avg_rate)
        if checkpoint:
            self.last_loss = avg_loss
        return avg_loss

    def close(self, path):
        self.writer.export_scalars_to_json(path)
        self.writer.close()
