import pickle
import random
import sys

import torch.utils.data
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from replication.ml.data.data_splitter import DataSplitter
from replication.ml.data.user_splitter import UserSplitter
from replication.ml.network.network import TouchNet
from replication.ml.trainer.trainer import TouchTrainer


def process_data(data_model: torch.nn.Module, data_loader: DataLoader, device: torch.device):
    result = []
    for data in data_loader:
        data = [tensor.to(device) for tensor in data]
        result += [data_model(data)]
    return torch.cat(result)


if __name__ == "__main__":
    random.seed(24)
    mode = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("./processed_data/users.pickle", "rb") as user_file:
        users = pickle.load(user_file)

    data_splitter = DataSplitter(users)

    if mode == "train":
        model = TouchNet(7, 112).to(device)
        trainer = TouchTrainer(
            model,
            data_splitter.train_loader,
            data_splitter.val_loader,
            data_splitter.test_loader,
            device
        )

        trainer.train(None)
        # trainer.close("./train.json")
    elif mode == "test":
        model = TouchNet(7, 112).to(device)
        trainer = TouchTrainer(
            model,
            data_splitter.train_loader,
            data_splitter.val_loader,
            data_splitter.test_loader,
            device
        )
        trainer.load_from(sys.argv[2])
        user_splitter = UserSplitter(users)
        user_id = 0
        model.eval()
        stressed_result = process_data(model, user_splitter.user_loaders[0][0], device)
        unstressed_result = process_data(model, user_splitter.user_loaders[0][1], device)
        stressed_result_np = stressed_result.cpu().detach().numpy()
        unstressed_result_np = unstressed_result.cpu().detach().numpy()

        sns.distplot(stressed_result_np, color="skyblue", label="Stressed")
        sns.distplot(unstressed_result_np, color="red", label="Unstressed")
        plt.legend()
        plt.savefig(f'user {user_id}.pdf')
