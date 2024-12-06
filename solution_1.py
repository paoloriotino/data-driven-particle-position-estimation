import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


df_dev = pd.read_csv('/DSL_Winter_Project_2024/development.csv')
df_eva = pd.read_csv('/DSL_Winter_Project_2024/evaluation.csv', index_col='Id')

df_dropped = df_dev.copy()
df_eva_dropped = df_eva.copy()


for i in range(18):
    df_dropped = df_dropped.drop(columns=[f'area[{i}]', f'rms[{i}]', f'tmax[{i}]'])
    df_eva_dropped = df_eva_dropped.drop(columns=[f'area[{i}]', f'rms[{i}]', f'tmax[{i}]'])

for i in [0, 7, 12, 16, 17]:
    df_dropped = df_dropped.drop(columns=[f'pmax[{i}]', f'negpmax[{i}]'])
    df_eva_dropped = df_eva_dropped.drop(columns=[f'pmax[{i}]', f'negpmax[{i}]'])

X = df_dropped.drop(columns=['x', 'y']).copy()
y = df_dev[['x', 'y']].copy()
X_test = df_eva_dropped.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

etr = ExtraTreesRegressor(max_depth=29, min_samples_split=2, n_estimators=700, random_state=42)

etr.fit(X_scaled, y)

y_eva_pred = etr.predict(X_test_scaled)
y_eva_pred = pd.DataFrame(y_eva_pred, columns=['x', 'y'])
y_eva_pred['index'] = range(len(y_eva_pred))
y_eva_pred['xy'] = y_eva_pred.apply(lambda row: '|'.join(map(str, row[['x', 'y']])), axis=1)
y_eva_pred = y_eva_pred.drop(['x', 'y', 'index'], axis=1)

y_eva_pred.to_csv("output_1.csv", index_label="Id", header=['Predicted'])
