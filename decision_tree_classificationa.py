# decision_tree_classificationa.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from jqdatasdk import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np

auth('18406425088', 'Aa12345678')  # JQData账号和密码

# 获取数据：假设df包含特征和目标变量（下一个交易日的开盘价）
df = get_price('000001.XSHE', end_date='2024-01-30 14:00:00', count=10000, frequency='daily', fields=['open', 'close', 'high', 'low', 'volume', 'money'])

# 增加p_change列
df['p_change'] = df['close'].pct_change().shift(-1)

# 填充缺失值（假设用均值填充）
df.fillna(df.mean(), inplace=True)

# 准备特征和目标变量
X_cls = df[['open', 'close', 'high', 'low', 'volume', 'money']]  # 分类任务特征
y_cls = (df['p_change'] > 0).astype(int)  # 分类任务目标变量，涨跌标签（1为涨，0为跌）

# 划分训练集和测试集（分类任务）
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# 初始化分类模型
clf = DecisionTreeClassifier(random_state=42)

# 训练分类模型
clf.fit(X_cls_train, y_cls_train)
y_cls_pred = clf.predict(X_cls_test)
accuracy = accuracy_score(y_cls_test, y_cls_pred)
print(f'分类模型精度为: {accuracy}')

##### 可视化部分

# 特征分布
plt.figure(figsize=(20, 15))
X_cls.hist(bins=50)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Feature Distribution')
plt.show()

# 目标变量分布
plt.figure(figsize=(10, 6))
sns.countplot(x=y_cls)
plt.xlabel('Target Variable')
plt.ylabel('Count')
plt.title('Target Variable Distribution')
plt.show()

# 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_cls_test, y_cls_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# ROC曲线
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_cls_test, clf.predict_proba(X_cls_test)[:, 1])
roc_auc = roc_auc_score(y_cls_test, clf.predict_proba(X_cls_test)[:, 1])
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# 特征重要性
plt.figure(figsize=(15, 5))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Feature Importances")
plt.bar(range(X_cls.shape[1]), importances[indices], align="center")
plt.xticks(range(X_cls.shape[1]), X_cls.columns[indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()
