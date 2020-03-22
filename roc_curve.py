#%%
import random
import pylab as pl
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn import preprocessing
import pandas
import json

percentages = [0, 10, 20, 30]

all_tpr = []
all_fpr = []
all_roc = []

all_user_roc = []

all_user_acc = []

user_names = ['B8',
 'A4',
 'A2',
 'B3',
 'A6',
 'B11',
 'A11',
 'B7',
 'A5',
 'B4',
 'A8',
 'B5',
 'A1',
 'B1',
 'A10',
 'B9',
 'B10',
 'B2']

for i in range(4):
    name = f"output_reference_{percentages[i]}.csv"
    #%%
    df_a = pandas.read_csv(f"./{name}")
    df_b = df_a.copy()

    #%%
    df_a.output = [json.loads(output)[0][0] for output in df_a.output]
    df_a['reference'] = [1 for reference in df_a.reference]
    df_b.output = [json.loads(output)[1][0] for output in df_b.output]
    df_b['reference'] = [0 for reference in df_b.reference]

    #%%
    df = pandas.concat((df_a, df_b))

    #%%
    pl.clf()
    user_roc = []
    user_acc = []
    for user in df.user.unique():
        user_df = df[df.user == user]
        fpr, tpr, _ = roc_curve(user_df.reference, user_df.output)
        acc = (tpr + (1 - fpr)) / 2
        max_acc = np.max(acc)
        roc = roc_auc_score(user_df.reference, user_df.output)
        user_roc += [roc]
        user_acc += [max_acc]
        pl.plot(fpr, tpr, label=f"{user_names[user]}: {roc:.2f}")
    all_user_roc += [user_roc]
    all_user_acc += [user_acc]
    fpr, tpr, _ = roc_curve(df.reference, df.output)
    roc = roc_auc_score(df.reference, df.output)
    all_tpr += [tpr]
    all_fpr += [fpr]
    all_roc += [roc]
    pl.plot(fpr, tpr, label=f"All: {roc:.2f}")
    pl.xlabel('False positive rate')
    pl.ylabel('True positive rate')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.legend(bbox_to_anchor=(1.0, 1.0))
    pl.title(f"{i * 10}% personalization data")
    # pl.show()
    pl.tight_layout()
    pl.savefig(f"{name}.png")

pl.clf()
for i in range(4):
    pl.plot(all_fpr[i], all_tpr[i], label=f"{percentages[i]}% personal: {all_roc[i]:.2f}")
pl.xlabel('False positive rate')
pl.ylabel('True positive rate')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.legend(bbox_to_anchor=(1.0, 1.0))
# pl.show()
pl.tight_layout()
pl.savefig(f"all.png")


#%%
import scipy.stats

stat_results = [["" for j in range(4)] for i in range(4)]

for i in range(4):
    for j in range(4):
        stat_result = scipy.stats.ttest_rel(all_user_roc[i], all_user_roc[j])
        stat_results[i][j] = f"{stat_result[0]:.2f}, {stat_result[1]:.2f}"

stat_results_np = np.array(stat_results)
print(stat_results_np)

for i in range(4):
    name = f"output_reference_{percentages[i]}.csv"
    #%%
    df = pandas.read_csv(f"./{name}")
    df['accuracy'] = [json.loads(output)[0][0] > json.loads(output)[1][0] for output in df.output]

all_user_acc_np = np.array(all_user_acc)

print(all_user_acc_np)

user_ids = []
model_ids = []
acc = []
for model_id, model in enumerate(all_user_acc):
    for user_id, user in enumerate(model):
        user_ids += [user_names[user_id]]
        model_ids += [f"{model_id * 10}%"]
        acc += [user]

all_user_acc_df = pandas.DataFrame({"User": user_ids, "Percentage": model_ids, "Accuracy": acc})

import seaborn as sns
import matplotlib.pyplot as plt
# sns.set()

plt.clf()
sns.barplot(x='User', y='Accuracy', data=all_user_acc_df[all_user_acc_df.Percentage == '10%'], palette="colorblind").set(title='Accuracy: 10% personalization')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"acc_10.png")

plt.clf()
sns.barplot(x='User', y='Accuracy', data=all_user_acc_df[all_user_acc_df.Percentage == '30%'], palette="colorblind").set(title='Accuracy: 30% personalization')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"acc_30.png")

plt.clf()
sns.barplot(x='Percentage', y='Accuracy', data=all_user_acc_df, palette="colorblind").set(title='Accuracy: personalization')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"acc.png")