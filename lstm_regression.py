# lstm_regression.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from jqdatasdk import *
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 认证并获取数据
auth('18406425088', 'Aa12345678')  # 账号是申请时所填写的手机号；密码为聚宽官网登录密码
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

# Reshape data for LSTM input: [samples, time steps, features]
# Assuming we use a window of 1 time step for simplicity
X_train_lstm = np.expand_dims(X_reg_train.values, axis=2)
X_test_lstm = np.expand_dims(X_reg_test.values, axis=2)

# 初始化 LSTM 模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练 LSTM 模型
history = model.fit(X_train_lstm, y_reg_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_reg_test), verbose=1)

# 使用 LSTM 模型进行预测
y_reg_pred_lstm = model.predict(X_test_lstm)

# 计算评估指标
mae_lstm = mean_absolute_error(y_reg_test, y_reg_pred_lstm)
mse_lstm = mean_squared_error(y_reg_test, y_reg_pred_lstm)
rmse_lstm = mean_squared_error(y_reg_test, y_reg_pred_lstm, squared=False)
r2_lstm = r2_score(y_reg_test, y_reg_pred_lstm)

# 输出评估结果
print(f'LSTM模型 MAE: {mae_lstm}')
print(f'LSTM模型 MSE: {mse_lstm}')
print(f'LSTM模型 RMSE: {rmse_lstm}')
print(f'LSTM模型 R^2: {r2_lstm}')

# 绘制训练历史
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练历史')
plt.xlabel('Epochs')
plt.ylabel('损失')
plt.legend()
plt.show()
