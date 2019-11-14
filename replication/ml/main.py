import pickle
import random
import sys
import json

import torch.utils.data

from replication.ml.data.data_splitter import DataSplitter
from replication.ml.network.network import TouchNet
from replication.ml.trainer.trainer import TouchTrainer

#%%
#
# with open("./processed_data/users.pickle", "rb") as user_file:
#     users = pickle.load(user_file)
#
# data_splitter = DataSplitter(users)
#
# #%%
#
# load_1 = data_splitter.train_loader
#
# #%%
# import matplotlib.pyplot as plt
#
# load_1.dataset.users[1].unstressed_condition.tasks[0].draw_cursors(plt.gca())
#
#
# plt.savefig("test.pdf")
# plt.clf()
# #%%
#

#%%
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
