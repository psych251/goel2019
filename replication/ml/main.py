import pickle
import random
import sys
import os

import numpy as np
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
    np.random.seed(24)
    torch.manual_seed(24)
    mode = sys.argv[1]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")

    with open("./processed_data/users_normalized.pickle", "rb") as user_file:
        users = pickle.load(user_file)

    data_splitter = DataSplitter(users)

    if mode == "train":
        trainer = TouchTrainer(
            data_splitter,
            device
        )

        if len(sys.argv) > 2:
            trainer.train(sys.argv[2])
        else:
            trainer.train(None)
        # trainer.close("./train.json")
    elif mode == "test":
        trainer = TouchTrainer(
            data_splitter,
            device
        )
        files = os.listdir(sys.argv[2])
        train_iter_limit = int(sys.argv[3]) if len(sys.argv) >= 4 else -1
        for file in files:
            name_segments = file.split('_')
            if len(name_segments) > 0 and name_segments[0] == 'modelTrained':
                file_name, user_name, train_iter_str, loss = name_segments
                train_iter = int(train_iter_str)
                if train_iter_limit == -1 or train_iter < train_iter_limit:
                    trainer.save_files[user_name][train_iter] = sys.argv[2] + "/" + file

        data_frame_dict = {"user": [], "output": [], "reference": []}
        for user_id, user in enumerate(trainer.user_names):
            trainer.current_user_id = user_id
            trainer.load_model(trainer.current_user_name)
            _, _, _, output, reference = trainer.eval(test=True)
            data_frame_dict["user"] += [user_id] * len(output)
            data_frame_dict["output"] += output.tolist()
            data_frame_dict["reference"] += reference.tolist()

        import pandas
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pandas.DataFrame(data_frame_dict)
        df.to_csv('output_reference_20200526T151100.csv')
        # sns.barplot('user', 'rate', data=df)
        # plt.savefig("user_rate.pdf")


        # user_splitter = UserSplitter(users)
        # for user_id in range(len(user_splitter.user_loaders)):
        #     model.eval()
        #     stressed_result = process_data(model, user_splitter.user_loaders[user_id][0], device)
        #     unstressed_result = process_data(model, user_splitter.user_loaders[user_id][1], device)
        #     stressed_result_np = stressed_result.cpu().detach().numpy()
        #     unstressed_result_np = unstressed_result.cpu().detach().numpy()
        #
        #     sns.distplot(stressed_result_np, color="skyblue", label="Stressed")
        #     sns.distplot(unstressed_result_np, color="red", label="Unstressed")
        #     plt.legend()
        #     plt.savefig(f'user {user_id}.pdf')
        #     plt.clf()
