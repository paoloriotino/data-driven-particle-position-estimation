import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler


df_dev = pd.read_csv('/DSL_Winter_Project_2024/development.csv')
df_eva = pd.read_csv('/DSL_Winter_Project_2024/evaluation.csv', index_col='Id')

df_dropped = df_dev.copy()
df_eva_dropped = df_eva.copy()
selected_features = ['pmax[8]','pmax[14]','negpmax[8]','area[1]','negpmax[5]','negpmax[10]','negpmax[14]','pmax[15]','negpmax[6]','pmax[2]','negpmax[9]','negpmax[4]','tmax[2]','tmax[5]','tmax[10]','pmax[4]','pmax[11]','pmax[3]','pmax[10]','pmax[5]','area[11]','area[3]','negpmax[3]','negpmax[11]','area[10]','area[4]','pmax[13]','area[5]','negpmax[13]','area[13]','pmax[6]','pmax[9]','tmax[4]','area[6]','area[9]','pmax[1]','tmax[11]']

df_dropped= df_dropped[selected_features]
df_eva_dropped = df_eva_dropped[selected_features]

X=df_dropped
y = df_dev[['x', 'y']].copy()
X_test = df_eva_dropped.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

etr = ExtraTreesRegressor(max_depth=29, max_features=1.0, min_samples_split=2, n_estimators=700, random_state=42)

etr.fit(X_scaled, y)

y_eva_pred = etr.predict(X_test_scaled)
y_eva_pred = pd.DataFrame(y_eva_pred, columns=['x', 'y'])
y_eva_pred['index'] = range(len(y_eva_pred))
y_eva_pred['xy'] = y_eva_pred.apply(lambda row: '|'.join(map(str, row[['x', 'y']])), axis=1)
y_eva_pred = y_eva_pred.drop(['x', 'y', 'index'], axis=1)

y_eva_pred.to_csv("output_2.csv", index_label="Id", header=['Predicted'])
