import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\lab7\ITA105_Lab_7.csv")

num_cols = df.select_dtypes(include=[np.number]).columns

skewness = df[num_cols].skew().sort_values(key=abs, ascending=False)

top10_skew = skewness.head(10)
print(top10_skew)

top3_cols = top10_skew.index[:3]

for col in top3_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

col_pos1 = 'LotArea'
col_pos2 = 'SalePrice'
col_neg = 'NegSkewIncome' 

df['log_'+col_pos1] = np.log(df[col_pos1])
df['log_'+col_pos2] = np.log(df[col_pos2])

df['boxcox_'+col_pos1], lambda1 = boxcox(df[col_pos1])
df['boxcox_'+col_pos2], lambda2 = boxcox(df[col_pos2])

pt = PowerTransformer(method='yeo-johnson')
df['power_'+col_neg] = pt.fit_transform(df[[col_neg]])

def plot_compare(col, transformed):
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    sns.histplot(df[col], kde=True)
    plt.title("Before")
    
    plt.subplot(1,2,2)
    sns.histplot(df[transformed], kde=True)
    plt.title("After")
    
    plt.show()

plot_compare(col_pos1, 'log_'+col_pos1)
plot_compare(col_pos1, 'boxcox_'+col_pos1)

result = pd.DataFrame({
    'Column': [col_pos1, col_pos2],
    'Skew_before': [df[col_pos1].skew(), df[col_pos2].skew()],
    'Skew_log': [df['log_'+col_pos1].skew(), df['log_'+col_pos2].skew()],
    'Skew_boxcox': [df['boxcox_'+col_pos1].skew(), df['boxcox_'+col_pos2].skew()]
})

print(result)


y = df['SalePrice']
X = df.drop('SalePrice', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'string']).columns

preprocessor_A = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

model_A = Pipeline([
    ('preprocess', preprocessor_A),
    ('regressor', LinearRegression())
])

model_A.fit(X_train, y_train)

pred_A = model_A.predict(X_test)

rmse_A = np.sqrt(mean_squared_error(y_test, pred_A))
r2_A = r2_score(y_test, pred_A)

# log target
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

model_B = Pipeline([
    ('preprocess', preprocessor_A),  # giống A
    ('regressor', LinearRegression())
])

model_B.fit(X_train, y_train_log)

pred_log = model_B.predict(X_test)

pred_B = np.exp(pred_log)

rmse_B = np.sqrt(mean_squared_error(y_test, pred_B))
r2_B = r2_score(y_test, pred_B)

preprocessor_C = ColumnTransformer([
    ('num', PowerTransformer(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

model_C = Pipeline([
    ('preprocess', preprocessor_C),
    ('regressor', LinearRegression())
])

model_C.fit(X_train, y_train)

pred_C = model_C.predict(X_test)

rmse_C = np.sqrt(mean_squared_error(y_test, pred_C))
r2_C = r2_score(y_test, pred_C)


result = pd.DataFrame({
    'Model': ['A - Raw', 'B - Log Target', 'C - Power Transform'],
    'RMSE': [rmse_A, rmse_B, rmse_C],
    'R2': [r2_A, r2_B, r2_C]
})

print(result)

col1 = 'SalePrice'
col2 = 'LotArea'

df['log_SalePrice'] = np.log(df['SalePrice'])
df['log_LotArea'] = np.log(df['LotArea'])

plt.figure()
sns.histplot(df['SalePrice'], kde=True)
plt.title("SalePrice - Raw")
plt.show()

plt.figure()
sns.histplot(df['LotArea'], kde=True)
plt.title("LotArea - Raw")
plt.show()

plt.figure()
sns.histplot(df['log_SalePrice'], kde=True)
plt.title("SalePrice - Log Transformed")
plt.show()

plt.figure()
sns.histplot(df['log_LotArea'], kde=True)
plt.title("LotArea - Log Transformed")
plt.show()