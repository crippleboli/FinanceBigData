# decision_tree_regression.py

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from jqdatasdk import *
import matplotlib.pyplot as plt
import numpy as np

auth('18406425088', 'Aa12345678')  # 账号是申请时所填写的手机号；密码为聚宽官网登录密码

# 获取数据，假设 df 包含了特征和目标变量（第二天的开盘价）
df = get_price('000001.XSHE', end_date='2024-01-30 14:00:00', count=10000, frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])

# 增加 p_change 列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值（假设用均值填充）
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X_reg = df[['open', 'close', 'high', 'low', 'volume', 'money']].copy()  # 使用 .copy() 避免 SettingWithCopyWarning
y_reg = df['open'].shift(-1)  # 回归任务目标变量，第二天的开盘价

# 使用插值方法填充 NaN 值
X_reg.interpolate(method='linear', inplace=True)
y_reg.interpolate(method='linear', inplace=True)

# 删除包含 NaN 值的行，确保 X_reg 和 y_reg 的行数一致
X_reg.dropna(inplace=True)
y_reg = y_reg[X_reg.index]  # 只保留 X_reg 对应的索引位置的 y_reg 数据

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
print(f'回归模型MAE: {mae}')
print(f'回归模型MSE: {mse}')
print(f'回归模型RMSE: {rmse}')
print(f'回归模型R^2: {r2}')




#####可视化部分

# 特征分布
X_reg.hist(bins=50, figsize=(20, 15))
plt.show()

# 目标变量分布
y_reg.hist(bins=50)
plt.show()

# 预测 vs 实际
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
plt.title("Feature Importances")
plt.bar(range(X_reg.shape[1]), importances[indices], align="center")
plt.xticks(range(X_reg.shape[1]), X_reg.columns[indices], rotation=90)
plt.show()

















