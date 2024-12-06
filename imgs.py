import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="white")


df_dev = pd.read_csv('/DSL_Winter_Project_2024/development.csv')
df_eva = pd.read_csv('/DSL_Winter_Project_2024/evaluation.csv', index_col='Id')

y = df_dev[['x', 'y']]

pmax_dataset = pd.concat([y, df_dev[[f'pmax[{i}]' for i in range(18)]]], axis=1)
negpmax_dataset = pd.concat([y, df_dev[[f'negpmax[{i}]' for i in range(18)]]], axis=1)
area_dataset = pd.concat([y, df_dev[[f'area[{i}]' for i in range(18)]]], axis=1)
tmax_dataset = pd.concat([y, df_dev[[f'tmax[{i}]' for i in range(18)]]], axis=1)
rms_dataset = pd.concat([y, df_dev[[f'rms[{i}]' for i in range(18)]]], axis=1)

corr_matrix_total = df_dev.corr().abs()
corr_matrix_pmax = pmax_dataset.corr().abs()
corr_matrix_negpmax = negpmax_dataset.corr().abs()
corr_matrix_area = area_dataset.corr().abs()
corr_matrix_tmax = tmax_dataset.corr().abs()
corr_matrix_rms = rms_dataset.corr().abs()

labels = ['x', 'y']
for i in range(18):
    labels.append(i)

all_corr_matrices = np.array([corr_matrix_pmax, corr_matrix_negpmax, corr_matrix_area, corr_matrix_tmax, corr_matrix_rms])
average_corr_values = np.mean(all_corr_matrices, axis=0)
average_corr_matrix_all_datasets = pd.DataFrame(average_corr_values, columns=labels, index=labels)

etr_min_split = [2, 3, 4, 5, 6, 7]
etr_distance_min_split = [4.312660487490908, 4.331340488222453, 4.326524952096753, 4.3407422534456455, 4.3356749070264184, 4.346409501988092]
etr_distance_min_split = [round(num, 3) for num in etr_distance_min_split]

rfr_min_split = [2, 3, 4, 5, 6, 7]
rfr_distance_min_split = [4.386210312596236, 4.396293224383761, 4.392321496096938, 4.392413180339165, 4.399279989955643, 4.40090758967779]
rfr_distance_min_split = [round(num, 3) for num in rfr_distance_min_split]

etr_max_depth = [15, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
etr_distance_max_depth = [6.900968881426223, 5.023943176699076, 4.504807551849427, 4.435377929686693, 4.380727680088782, 4.344892925674776, 4.312660487490908, 4.296665134861084, 4.270385619056734, 4.27394582437725, 4.270228047327625, 4.252772858255733, 4.24975432145236]
etr_distance_max_depth = [round(num, 3) for num in etr_distance_max_depth]

rfr_max_depth = [15, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
rfr_distance_max_depth = [4.955448704446334, 4.469000812790143, 4.4016079888269966, 4.39961316743399, 4.400152593201356, 4.396285476004664, 4.386210312596236, 4.39444604581652, 4.392281595932754, 4.393586122852191, 4.388442245689659, 4.402148715951128, 4.394750195169761]
rfr_distance_max_depth = [round(num, 3) for num in rfr_distance_max_depth]

fontsize = 18
labelsize = 15

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.lineplot(x=etr_min_split, y=etr_distance_min_split, ax=ax1, label='Line 1', color='b')
ax1.set_xlabel('ETR min_samples_split values', color='b', labelpad=10, fontsize=fontsize)
ax1.set_ylabel('distances', labelpad=10, fontsize=fontsize)
ax1.tick_params(axis='x', colors='b', labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.legend().remove()
ax1_twin = ax1.twiny()
ax1_twin.xaxis.tick_top()
ax1_twin.xaxis.set_label_position("top")
ax1_twin.set_xlabel('RFR min_samples_split values', color='r', labelpad=10, fontsize=fontsize)
ax1_twin.tick_params(axis='x', colors='r', labelsize=labelsize)
ax1_twin.tick_params(axis='y', labelsize=labelsize)
ax1.yaxis.set_ticks_position('left')
sns.lineplot(x=rfr_min_split, y=rfr_distance_min_split, ax=ax1_twin, label='Line 2', color='r')
ax1_twin.legend().remove()
ax1.grid(True, alpha=.6)

sns.lineplot(x=etr_max_depth, y=etr_distance_max_depth, ax=ax2, label='Line 1', color='b')
ax2.set_xlabel('ETR min_samples_split values', color='b', labelpad=10, fontsize=fontsize)
ax2.set_ylabel('distances', labelpad=10, fontsize=fontsize)
ax2.tick_params(axis='x', colors='b', labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.set_xticks(np.arange(15, 34, 2))
ax2.legend().remove()
ax2_twin = ax2.twiny()
ax2_twin.xaxis.tick_top()
ax2_twin.xaxis.set_label_position("top")
ax2_twin.set_xlabel('RFR min_samples_split values', color='r', labelpad=10, fontsize=fontsize)
ax2_twin.tick_params(axis='x', colors='r', labelsize=labelsize)
ax2_twin.tick_params(axis='y', labelsize=labelsize)
ax2.yaxis.set_ticks_position('left')
ax2_twin.set_xticks(np.arange(15, 34, 2))
sns.lineplot(x=rfr_max_depth, y=rfr_distance_max_depth, ax=ax2_twin, label='Line 2', color='r')
ax2_twin.legend().remove()
ax2.grid(True, alpha=.6)

plt.subplots_adjust(left=0.12, right=0.95, top=0.83, bottom=0.17)
plt.tight_layout()
plt.show()
fig.savefig("etrVSrfr.pdf", bbox_inches="tight")


fig, ax = plt.subplots(1, 2, figsize=(22, 10))
heatmap1 = sns.heatmap(corr_matrix_pmax, cmap='coolwarm', fmt=".2f", ax=ax[0], cbar=False, linewidths=.5)
ax[0].set_title('Target-pmax correlation matrix', fontsize=25, y=1.015)
ax[0].tick_params(axis='both', which='both', labelsize=18)
heatmap1.set_aspect('equal')
heatmap2 = sns.heatmap(average_corr_matrix_all_datasets, cmap='coolwarm', fmt=".2f", ax=ax[1], linewidths=.5)
ax[1].set_title("Mean pad-wise correlation matrix", fontsize=25, y=1.015)
ax[1].tick_params(axis='both', which='both', labelsize=18)
heatmap2.set_aspect('equal')
plt.tight_layout()
plt.show()
fig.savefig("corr.pdf", bbox_inches="tight")