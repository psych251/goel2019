import os
from typing import List

from replication.common import TaskType
from replication.preprocess.user import User

DATA_DIR = "./original_data/data/raw_data/"


def load_data(dir=DATA_DIR) -> List[User]:
    user_list = []
    user_dir_names = os.listdir(dir)
    for dir_name in user_dir_names:
        user_dir = os.path.join(dir, dir_name)
        user = User(user_dir, dir_name)
        user.clean_data()
        user_list += [user]
    return user_list


if __name__ == '__main__':
# %%
    users = load_data()


    # %%
    all_tasks_data_count = []
    for user in users:
        for tasks in user.stressed_condition.tasks:
            all_tasks_data_count += [len(tasks.cursor_entries)]
        for tasks in user.unstressed_condition.tasks:
            all_tasks_data_count += [len(tasks.cursor_entries)]

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.distplot(all_tasks_data_count)
    plt.show()

    #%%
    import pickle

    with open("./processed_data/users_normalized.pickle", "wb") as user_file:
        pickle.dump(users, user_file)

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

    #%%
    df = user.unstressed_condition.tasks[0].separated_track_pad_df
    df2 = df.reset_index(level=0)
    plt.figure(figsize=(5, 3))
    sns.scatterplot(x='time', y='x', hue='level_0', data=df2, linewidth=0, alpha=0.7)
    plt.savefig('test.pdf')