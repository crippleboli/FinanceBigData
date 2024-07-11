from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from jqdatasdk import *
import pandas as pd

auth('18406425088','Aa12345678')  # 账号是申请时所填写的手机号；密码为聚宽官网登录密码

# 获取数据，假设 df 包含了特征和目标变量（第二天的开盘价）
df = get_price('000001.XSHE', end_date='2024-01-30 14:00:00', count=10000, frequency='daily', fields=['open','close','high','low','volume','money'])

# 增加 p_change 列
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
