import logging
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from jqdatasdk import auth, get_price
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置日志
logging.basicConfig(level=logging.INFO, filename='rf_model_training.log', filemode='w')
logger = logging.getLogger(__name__)

# 认证并获取数据
auth('18406425088', 'Aa12345678')
df = get_price('000001.XSHE', start_date='1980-01-30 14:00:00', end_date='2024-01-30 14:00:00', frequency='daily',
                fields=['open', 'close', 'high', 'low', 'volume', 'money'])

# 增加 p_change 列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X_reg = df[['open', 'close', 'high', 'low', 'volume', 'money']].copy()
y_reg = df['open'].shift(-1)

# 使用插值方法填充 NaN 值
X_reg.interpolate(method='linear', inplace=True)
y_reg.interpolate(method='linear', inplace=True)

# 删除包含 NaN 值的行，确保 X_reg 和 y_reg 的行数一致
X_reg.dropna(inplace=True)
y_reg = y_reg[X_reg.index]

# 数据标准化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_reg = scaler_X.fit_transform(X_reg)
y_reg = scaler_y.fit_transform(y_reg.values.reshape(-1, 1))

# 按时间顺序分割数据
train_size = int(len(X_reg) * 0.9)
X_reg_train, X_reg_test = X_reg[:train_size], X_reg[train_size:]
y_reg_train, y_reg_test = y_reg[:train_size], y_reg[train_size:]

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_reg_train, y_reg_train.ravel())
y_reg_pred_rf = rf_model.predict(X_reg_test)
y_reg_pred_rf = scaler_y.inverse_transform(y_reg_pred_rf.reshape(-1, 1))

# 计算随机森林评估指标
y_reg_test_inv = scaler_y.inverse_transform(y_reg_test)
mae_rf = mean_absolute_error(y_reg_test_inv, y_reg_pred_rf)
mse_rf = mean_squared_error(y_reg_test_inv, y_reg_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_reg_test_inv, y_reg_pred_rf)

# 输出随机森林评估结果
logger.info(f'随机森林模型 MAE: {mae_rf}')
logger.info(f'随机森林模型 MSE: {mse_rf}')
logger.info(f'随机森林模型 RMSE: {rmse_rf}')
logger.info(f'随机森林模型 R^2: {r2_rf}')

# 绘制预测值与实际值比较图
plt.figure(figsize=(12, 6))
plt.plot(y_reg_test_inv, label='实际值')
plt.plot(y_reg_pred_rf, label='预测值')
plt.title('随机森林模型预测 vs 实际')
plt.xlabel('样本')
plt.ylabel('开盘价')
plt.legend()
plt.savefig('rf_prediction_vs_actual.png')
plt.show()

# 特征分布
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 2, i)
    plt.hist(df[column], bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{column} Distribution')
plt.tight_layout()
plt.savefig('feature_distribution.png')
plt.show()

# 目标变量分布
plt.figure(figsize=(10, 6))
plt.hist(df['open'], bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Target Variable Distribution')
plt.savefig('target_variable_distribution.png')
plt.show()

# 残差图
residuals = y_reg_test_inv - y_reg_pred_rf
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_pred_rf, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.savefig('residuals_vs_predicted.png')
plt.show()

# 特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(15, 5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [df.columns[i] for i in indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title("Feature Importances")
plt.savefig('feature_importances.png')
plt.show()

# 特征相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.savefig('feature_correlation_matrix.png')
plt.show()
