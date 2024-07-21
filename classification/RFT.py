from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jqdatasdk import auth, get_price
import numpy as np

# 认证
auth('16670149521', 'Wzy20030119')

# 获取数据
df = get_price('000001.XSHE', start_date='2023-01-30 14:00', end_date='2024-01-30 14:00', frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])
print(df.head())  # 检查数据是否成功获取

# 增加p_change列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值（用均值填充）
df.fillna(df.mean(), inplace=True)
print(df.head())  # 检查数据处理后的结果

# 准备特征和目标变量
X_cls = df[['open', 'close', 'high', 'low', 'volume', 'money']]
y_cls = (df['p_change'] > 0).astype(int)
print(X_cls.head())  # 检查特征变量
print(y_cls.head())  # 检查目标变量

# 设置窗口参数
min_window_size = 60
max_window_size = 120
prediction_days = 1  # 每个窗口预测一天

# 存储每个窗口大小的评估结果
best_window_size = None
best_accuracy = 0
best_roc_auc = 0
results = []

# 滑动窗口搜索最佳窗口大小
for window_size in range(min_window_size, max_window_size + 1):
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    print(f"Processing window size: {window_size}")

    for i in range(window_size, len(X_cls) - prediction_days + 1):
        X_train = X_cls.iloc[i - window_size:i]
        y_train = y_cls.iloc[i - window_size:i]
        X_test = X_cls.iloc[i:i + prediction_days]
        y_test = y_cls.iloc[i:i + prediction_days]

        # 初始化模型
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # 训练模型
        rf.fit(X_train, y_train)

        # 预测结果
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]

        # 存储结果
        y_true_all.extend(y_test.values)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # 转换为numpy数组
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    # 计算准确率和ROC AUC
    accuracy = accuracy_score(y_true_all, y_pred_all)
    roc_auc = roc_auc_score(y_true_all, y_prob_all)
    results.append((window_size, accuracy, roc_auc))

    print(f"Window size {window_size} - Accuracy: {accuracy}, ROC AUC: {roc_auc}")

    if roc_auc > best_roc_auc:
        best_window_size = window_size
        best_accuracy = accuracy
        best_roc_auc = roc_auc

print(f'最佳窗口大小: {best_window_size}')
print(f'最佳窗口大小下的RandomForest Accuracy: {best_accuracy}')
print(f'最佳窗口大小下的RandomForest ROC AUC: {best_roc_auc}')

# 结果可视化
results_df = pd.DataFrame(results, columns=['Window Size', 'Accuracy', 'ROC AUC'])

plt.figure(figsize=(10, 5))
plt.plot(results_df['Window Size'], results_df['Accuracy'], label='Accuracy', marker='o')
plt.plot(results_df['Window Size'], results_df['ROC AUC'], label='ROC AUC', marker='x')
plt.xlabel('Window Size')
plt.ylabel('Score')
plt.title('Window Size vs Model Performance')
plt.legend()
plt.show()

# 使用最佳窗口大小重新训练和评估模型
y_true_all = []
y_pred_all = []
y_prob_all = []

for i in range(best_window_size, len(X_cls) - prediction_days + 1):
    X_train = X_cls.iloc[i - best_window_size:i]
    y_train = y_cls.iloc[i - best_window_size:i]
    X_test = X_cls.iloc[i:i + prediction_days]
    y_test = y_cls.iloc[i:i + prediction_days]

    # 初始化模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf.fit(X_train, y_train)

    # 预测结果
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # 存储结果
    y_true_all.extend(y_test.values)
    y_pred_all.extend(y_pred)
    y_prob_all.extend(y_prob)

# 转换为numpy数组
y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)
y_prob_all = np.array(y_prob_all)

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
plt.plot(fpr, tpr, label=f'ROC curve (area = {best_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 特征重要性
plt.figure(figsize=(15, 5))
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Feature Importances")
plt.bar(range(X_cls.shape[1]), importances[indices], align="center")
plt.xticks(range(X_cls.shape[1]), X_cls.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
