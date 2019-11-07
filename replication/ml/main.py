import pickle
import random
import sys
import json

import torch.utils.data

from replication.ml.data.data_splitter import DataSplitter
from replication.ml.network.network import TouchNet
from replication.ml.trainer.trainer import TouchTrainer

if __name__ == "__main__":
    random.seed(24)
    mode = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("./processed_data/users.pickle", "rb") as user_file:
        users = pickle.load(user_file)

    data_splitter = DataSplitter(users)

    if mode == "train":
        model = TouchNet(2, 32)
        trainer = TouchTrainer(
            model,
            data_splitter.train_loader,
            data_splitter.val_loader,
            data_splitter.test_loader,
            device
        )

        trainer.train(None)
        # trainer.close("./train.json")
