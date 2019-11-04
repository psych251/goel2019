import os
from typing import List

from replication.common import TaskType
from replication.preprocess.user import User

DATA_DIR = "./original_data/data/raw_data/"


def load_data() -> List[User]:
    user_list = []
    user_dir_names = os.listdir(DATA_DIR)
    for dir_name in user_dir_names:
        user_dir = os.path.join(DATA_DIR, dir_name)
        user_list += [User(user_dir)]
    return user_list


# if __name__ == '__main__':
# %%
users = load_data()

# %%
stressed_means = []
unstressed_means = []

for user in users:
    for task in user.stressed_condition.tasks:
        if task.task_type == TaskType.CLICK:
            stressed_means += [task.track_pad_df.contact_area.mean()]

    for task in user.unstressed_condition.tasks:
        if task.task_type == TaskType.CLICK:
            unstressed_means += [task.track_pad_df.contact_area.mean()]

# %%
import numpy as np

stressed_means_np = np.array(stressed_means)
unstressed_means_np = np.array(unstressed_means)

filtered_stressed_means_np = stressed_means_np[~np.isnan(stressed_means_np)]
filtered_unstressed_means_np = unstressed_means_np[~np.isnan(unstressed_means_np)] 

#%%
import pandas

stressed_count = len(filtered_stressed_means_np)
unstressed_count = len(filtered_unstressed_means_np)
df = pandas.DataFrame({
    'condition':
        ["stressed"] * stressed_count +
        ["unstressed"] * unstressed_count,
    'average area':
        filtered_stressed_means_np.tolist() +
        filtered_unstressed_means_np.tolist()
})

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.figure(figsize=(5, 3))

plot = sns.barplot(x='condition', y='average area', data=df)

plt.savefig('average area.pdf')

