import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from jqdatasdk import auth, get_price

# 认证
auth('18406425088', 'Aa12345678')

# 获取数据
df = get_price('000001.XSHE', start_date='2022-01-30 14:00', end_date='2024-01-30 14:00:00', frequency='daily',
               fields=['open', 'close', 'high', 'low', 'volume', 'money'])
df['p_change'] = df['close'].pct_change().shift(-1)
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X = df[['open', 'close', 'high', 'low', 'volume', 'money']]
y = (df['p_change'] > 0).astype(int)

# 设置窗口参数
window_size = 60  # 训练集大小
prediction_days = 1  # 每个窗口预测一天

# 模型初始化
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# 存储所有模型的预测结果
results = {name: {'y_true': [], 'y_pred': []} for name in models.keys()}

# 滑动窗口
for i in range(window_size, len(X) - prediction_days + 1):
    X_train = X.iloc[i - window_size:i]
    y_train = y.iloc[i - window_size:i]
    X_test = X.iloc[i:i + prediction_days]
    y_test = y.iloc[i:i + prediction_days]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 存储预测结果
        results[name]['y_true'].extend(y_test.values)
        results[name]['y_pred'].extend(y_pred)

# 计算准确率和ROC AUC
summary_results = {}
for name, data in results.items():
    y_true_combined = np.array(data['y_true'])
    y_pred_combined = np.array(data['y_pred'])
    accuracy = accuracy_score(y_true_combined, y_pred_combined)
    roc_auc = roc_auc_score(y_true_combined, y_pred_combined)
    summary_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} ROC AUC: {roc_auc}')

# 比较分类效果
results_df = pd.DataFrame(summary_results).T
print("Model Comparison Results:")
print(results_df)

# 选取最优分类器
best_model_name = results_df['roc_auc'].idxmax()
best_model = models[best_model_name]
print(f'最优分类器是: {best_model_name}')

# 可视化
plt.figure(figsize=(10, 5))
results_df['accuracy'].plot(kind='bar', color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(10, 5))
results_df['roc_auc'].plot(kind='bar', color='lightgreen')
plt.title('Model ROC AUC Comparison')
plt.ylabel('ROC AUC')
plt.show()
