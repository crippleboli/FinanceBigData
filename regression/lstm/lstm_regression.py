import logging
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from jqdatasdk import auth, get_price
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras_tuner import HyperModel, RandomSearch
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置日志
logging.basicConfig(level=logging.INFO, filename='lstm_model_training.log', filemode='w')
logger = logging.getLogger(__name__)

# 认证并获取数据
auth('18406425088', 'Aa12345678')
df = get_price('000001.XSHE', start_date='2000-01-30 14:00:00', end_date='2024-01-30 14:00:00', frequency='daily',
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

# 按时间顺序划分训练集和测试集
split_index = int(len(X_reg) * 0.9)
X_reg_train, X_reg_test = X_reg[:split_index], X_reg[split_index:]
y_reg_train, y_reg_test = y_reg[:split_index], y_reg[split_index:]

# Reshape data for LSTM input: [samples, time steps, features]
X_train_lstm = np.expand_dims(X_reg_train, axis=1)
X_test_lstm = np.expand_dims(X_reg_test, axis=1)

# 定义超参数模型
class LSTMHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 1, 3)):
            if i == 0:
                model.add(LSTM(
                    units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation_' + str(i), ['relu', 'tanh']),
                    input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]),
                    return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False
                ))
            else:
                model.add(LSTM(
                    units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation_' + str(i), ['relu', 'tanh']),
                    return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False
                ))
            model.add(Dropout(hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(
            optimizer=hp.Choice('optimizer', ['adam', 'nadam']),
            loss='mse',
            metrics=['mae']
        )
        return model

hypermodel = LSTMHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=1,
    directory='lstm_tuner_optimized',
    project_name='lstm_hyperparameter_tuning_optimized'
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
]

# 搜索超参数
tuner.search(
    X_train_lstm, y_reg_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_lstm, y_reg_test),
    callbacks=callbacks,
    verbose=1
)

# 获取最佳模型
best_model = tuner.get_best_models(num_models=1)[0]

# 使用最佳模型进行预测
logger.info("模型训练完成，开始预测...")
y_reg_pred_lstm = best_model.predict(X_test_lstm)
y_reg_pred_lstm = scaler_y.inverse_transform(y_reg_pred_lstm)

# 计算评估指标
y_reg_test_inv = scaler_y.inverse_transform(y_reg_test)
mae_lstm = mean_absolute_error(y_reg_test_inv, y_reg_pred_lstm)
mse_lstm = mean_squared_error(y_reg_test_inv, y_reg_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_reg_test_inv, y_reg_pred_lstm)

# 输出评估结果
logger.info(f'LSTM模型 MAE: {mae_lstm}')
logger.info(f'LSTM模型 MSE: {mse_lstm}')
logger.info(f'LSTM模型 RMSE: {rmse_lstm}')
logger.info(f'LSTM模型 R^2: {r2_lstm}')

# 获取最佳试验的历史记录
best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

# 提取历史数据
loss_history = best_trial.metrics.get_history('loss')
val_loss_history = best_trial.metrics.get_history('val_loss')

# 绘制预测值与实际值比较图
plt.figure(figsize=(12, 6))
plt.plot(y_reg_test_inv, label='实际值')
plt.plot(y_reg_pred_lstm, label='预测值')
plt.title('LSTM模型预测 vs 实际')
plt.xlabel('样本')
plt.ylabel('开盘价')
plt.legend()
plt.savefig('LSTM_prediction_vs_actual.png')
plt.show()

# 特征分布
plt.figure(figsize=(20, 15))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 2, i)
    plt.hist(df[column], bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{column} 分布')
plt.tight_layout()
plt.savefig('feature_distribution.png')
plt.show()

# 目标变量分布
plt.figure(figsize=(10, 6))
plt.hist(df['open'], bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('目标变量分布')
plt.savefig('target_variable_distribution.png')
plt.show()

# 残差图
residuals = y_reg_test_inv - y_reg_pred_lstm
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_pred_lstm, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差 vs 预测值')
plt.savefig('residuals_vs_predicted.png')
plt.show()

# 特征重要性
importances = np.abs(best_model.layers[0].get_weights()[0]).sum(axis=1)  # 简单地用输入层权重绝对值之和衡量特征重要性
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(15, 5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [df.columns[i] for i in indices], rotation=90)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title("特征重要性")
plt.savefig('feature_importances.png')
plt.show()

# 特征相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('特征相关性矩阵')
plt.savefig('feature_correlation_matrix.png')
plt.show()
