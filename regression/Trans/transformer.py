import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jqdatasdk import auth, get_price
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
import seaborn as sns
import tensorflow as tf

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 认证并获取数据
auth('18406425088', 'Aa12345678')
df = get_price('000001.XSHE', start_date='1980-01-30 14:00', end_date='2024-04-14 14:00', frequency='daily',
               fields=['open', 'close', 'high', 'low', 'volume', 'money'])

# 增加技术指标
df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

# 增加 p_change 列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X_reg = df[['open', 'close', 'high', 'low', 'volume', 'money'] + [col for col in df.columns if
                                                                  'volatility' in col or 'trend' in col]].copy()
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

# 手动划分训练集和测试集，确保时间顺序
split_index = int(len(X_reg) * 0.9)  # 使用 90% 的数据作为训练集，10% 作为测试集
X_reg_train, X_reg_test = X_reg[:split_index], X_reg[split_index:]
y_reg_train, y_reg_test = y_reg[:split_index], y_reg[split_index:]

# Reshape data for Transformer input: [samples, time steps, features]
X_train_transformer = np.expand_dims(X_reg_train, axis=1)
X_test_transformer = np.expand_dims(X_reg_test, axis=1)

# 定义学习率调度器
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# 定义 Transformer 模型
def build_transformer_model(input_shape, learning_rate=0.001, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

input_shape = (X_train_transformer.shape[1], X_train_transformer.shape[2])
transformer_model = build_transformer_model(input_shape)

# 设置回调
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
    ModelCheckpoint(filepath='best_transformer_model.h5', monitor='val_loss', save_best_only=True),
    LearningRateScheduler(lr_schedule)
]

# 训练模型
history = transformer_model.fit(
    X_train_transformer, y_reg_train,
    validation_data=(X_test_transformer, y_reg_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# 使用最佳模型进行预测
logger.info("模型训练完成，开始预测...")
y_reg_pred_transformer = transformer_model.predict(X_test_transformer)
y_reg_pred_transformer = scaler_y.inverse_transform(y_reg_pred_transformer)

# 计算评估指标
y_reg_test_inv = scaler_y.inverse_transform(y_reg_test)
mae_transformer = mean_absolute_error(y_reg_test_inv, y_reg_pred_transformer)
mse_transformer = mean_squared_error(y_reg_test_inv, y_reg_pred_transformer)
rmse_transformer = np.sqrt(mse_transformer)
r2_transformer = r2_score(y_reg_test_inv, y_reg_pred_transformer)

# 输出评估结果
logger.info(f'Transformer模型 MAE: {mae_transformer}')
logger.info(f'Transformer模型 MSE: {mse_transformer}')
logger.info(f'Transformer模型 RMSE: {rmse_transformer}')
logger.info(f'Transformer模型 R^2: {r2_transformer}')

# 绘制预测值与实际值比较图
plt.figure(figsize=(12, 6))
plt.plot(y_reg_test_inv, label='实际值')
plt.plot(y_reg_pred_transformer, label='预测值')
plt.title('Transformer模型预测 vs 实际')
plt.xlabel('样本')
plt.ylabel('开盘价')
plt.legend()
plt.savefig('transformer_prediction_vs_actual.png')
plt.show()

# 绘制训练和验证损失
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.savefig('transformer_loss_curve.png')
plt.show()

# 绘制训练和验证MAE
plt.figure(figsize=(12, 6))
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('MAE曲线')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig('transformer_mae_curve.png')
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
residuals = y_reg_test_inv - y_reg_pred_transformer
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_pred_transformer, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.savefig('residuals_vs_predicted.png')
plt.show()
