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
df = get_price('000001.XSHE', start_date='2023-01-30 14:00', end_date='2024-01-30 14:00:00', frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])
df['p_change'] = df['close'].pct_change().shift(-1)
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X = df[['open', 'close', 'high', 'low', 'volume', 'money']]
y = (df['p_change'] > 0).astype(int)

# 手动按时间顺序划分训练集和测试集
split_index = int(len(X) * 0.8)  # 使用80%的数据作为训练集
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 模型初始化
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# 训练和评估模型
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test))
    results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    print(f'{name} Accuracy: {accuracy}')
    print(f'{name} ROC AUC: {roc_auc}')

# 比较分类效果
results_df = pd.DataFrame(results).T
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

# 最优分类器的详细评估
y_pred_best = best_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'{best_model_name} Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test))
roc_auc_best = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{best_model_name} Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
