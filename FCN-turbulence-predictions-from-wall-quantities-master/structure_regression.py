import pandas
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from sklearn import preprocessing

# df_init = pandas.read_csv("structure_stats_abs_inputs_yp15.csv")
# df_init = pandas.read_csv("structure_stats_abs_inputs_yp15_top20.csv")
df_init = pandas.read_csv("structure_stats_abs_inputs_yp15_top10_min10.csv")
uvw = 0
channel = 0

# select only those samples where uvw = 0 and channel=0
df = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 20)]

X = df[['area', 'length', 'height', 'original_sum']].to_numpy()
y = df['shap_abs_sum'].to_numpy()
# X = np.concatenate()
# add another column of ones to X
scaler = preprocessing.StandardScaler()
# X = scaler.fit_transform(X)
# no need to manually add the linear term since it's already included as intercept
# X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

regr = linear_model.LinearRegression()
regr.fit(X, y)

# Predict on the testing set
y_pred = regr.predict(X)

# Calculate R-squared score
r_squared = r2_score(y, y_pred)
coefs = np.concatenate((regr.coef_, np.array(regr.intercept_)[None]))
# concatenate the coefs with intercept in one array



print(regr.coef_)