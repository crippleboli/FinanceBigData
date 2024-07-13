# decision_tree_regression.py

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from jqdatasdk import *
import matplotlib.pyplot as plt
import numpy as np

auth('18406425088', 'Aa12345678')  # JQData账号和密码

# 获取数据：假设df包含特征和目标变量（下一个交易日的开盘价）
df = get_price('000001.XSHE', end_date='2024-01-30 14:00:00', count=10000, frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])

# 添加p_change列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值（假设用均值填充）
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X_reg = df[['open', 'close', 'high', 'low', 'volume', 'money']].copy()  # 使用.copy()避免SettingWithCopyWarning
y_reg = df['open'].shift(-1)  # 回归任务目标变量：下一个交易日的开盘价

# 使用线性插值填充NaN值
X_reg.interpolate(method='linear', inplace=True)
y_reg.interpolate(method='linear', inplace=True)

# 删除包含NaN值的行，确保X_reg和y_reg行数一致
X_reg.dropna(inplace=True)
y_reg = y_reg[X_reg.index]  # 仅保留与X_reg对应的索引位置的y_reg数据

# 划分训练集和测试集（回归任务）
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# 初始化回归模型
regressor = DecisionTreeRegressor(random_state=42)

# 训练回归模型
regressor.fit(X_reg_train, y_reg_train)
y_reg_pred = regressor.predict(X_reg_test)

# 计算回归模型评估指标
mae = mean_absolute_error(y_reg_test, y_reg_pred)
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = mean_squared_error(y_reg_test, y_reg_pred, squared=False)
r2 = r2_score(y_reg_test, y_reg_pred)

# 输出评估结果
print(f'Regression Model MAE: {mae}')
print(f'Regression Model MSE: {mse}')
print(f'Regression Model RMSE: {rmse}')
print(f'Regression Model R^2: {r2}')

##### Visualization Section

# 特征分布
plt.figure(figsize=(20, 15))
X_reg.hist(bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Feature Distribution')
plt.show()

# 目标变量分布
plt.figure(figsize=(10, 6))
y_reg.hist(bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Target Variable Distribution')
plt.show()

# 实际 vs 预测
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test, y_reg_pred)
plt.plot([min(y_reg_test), max(y_reg_test)], [min(y_reg_test), max(y_reg_test)], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# 残差图
residuals = y_reg_test - y_reg_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# 特征重要性
importances = regressor.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(15, 5))
plt.bar(range(X_reg.shape[1]), importances[indices], align="center")
plt.xticks(range(X_reg.shape[1]), X_reg.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title("Feature Importances")
plt.show()
