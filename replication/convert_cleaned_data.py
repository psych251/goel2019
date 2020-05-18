import json
import os
import re

import copy

from replication.common import TaskType, SAMPLE_INTERVAL
from replication.preprocess.condition import TaskMoves
from replication.preprocess.console import Task
from replication.preprocess.moves import TrackPadEntry
from replication.preprocess.user import User

CLEANED_DATA_DIR = "./processed_data/cleaned_data"


def parse_task_name(name: str):
    split_str = re.split(r'^(click|drag|steer)(\d+)px(\d+)px$', name)
    assert (len(split_str) == 5)
    type_str = split_str[1]
    width_str = split_str[2]
    height_str = split_str[3]
    return Task(name, TaskType.from_str(type_str), int(width_str), int(height_str))


def convert_moves(old_moves):
    new_moves = []
    for task_name, task_traces in old_moves.items():
        parsed_task = parse_task_name(task_name)
        # assert (len(task_traces) == 2)
        for task_trace in task_traces:
            if len(task_trace) == 0:
                continue
            timestamps = [entry['timestamp'] for stroke in task_trace for entry in stroke]
            time_start = min(timestamps)
            time_end = max(timestamps)
            current_task = copy.copy(parsed_task)
            current_task.start = time_start
            current_task.finish = time_end
            moves = TaskMoves(current_task)
            new_separated_track_pad_entries = [
                [
                    TrackPadEntry(
                        entry['timestamp'],
                        0,
                        entry['finger_state'],
                        entry['x'], entry['y'],
                        0, 0,
                        entry['major_axis'], entry['minor_axis'],
                        0
                    )
                    for entry in stroke
                ]
                for stroke in task_trace
            ]
            moves.separated_track_pad_entries = \
                TaskMoves.fill_separated_data_entries(new_separated_track_pad_entries, SAMPLE_INTERVAL)
            new_moves += [moves]
    return new_moves


# %%
user_list = []
user_dir_names = os.listdir(CLEANED_DATA_DIR)

# %%
users = []
for dir_name in user_dir_names:
    if '.' in dir_name:
        continue

    print(f"Working on {dir_name}...")
    user_dir = os.path.join(CLEANED_DATA_DIR, dir_name)
    stressed_cut_moves_dir = os.path.join(user_dir, "stressed_cut_moves.json")
    unstressed_cut_moves_dir = os.path.join(user_dir, "unstressed_cut_moves.json")

    with open(stressed_cut_moves_dir) as stressed_cut_moves_file:
        stressed_cut_moves = json.load(stressed_cut_moves_file)

    with open(unstressed_cut_moves_dir) as unstressed_cut_moves_file:
        unstressed_cut_moves = json.load(unstressed_cut_moves_file)
    current_user = User()
    current_user.name = dir_name
    current_user.stressed_condition.tasks = sorted(convert_moves(stressed_cut_moves), key=lambda move: move.start)
    current_user.unstressed_condition.tasks = sorted(convert_moves(unstressed_cut_moves), key=lambda move: move.start)
    current_user.normalize_separated_track_pad_entries()
    users += [current_user]

# %%
import pickle

with open("./processed_data/users_cleaned_normalized.pickle", "wb") as user_file:
    pickle.dump(users, user_file)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

df = users[0].unstressed_condition.tasks[100].separated_track_pad_df
df2 = df.reset_index(level=0)
plt.figure(figsize=(5, 3))
sns.scatterplot(x='time', y='x', hue='bundle', style='valid', data=df2, linewidth=0, alpha=0.7)
plt.savefig('test.pdf')
# plt.show()
