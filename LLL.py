import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.vision.transforms import ToTensor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, 
                           roc_auc_score, f1_score, 
                           classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('cleaned_credit_risk_dataset_processed.csv')

# 准备特征和目标变量
X = data.drop('loan_status', axis=1).values
y = data['loan_status'].values.astype('float32')

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 转换为Paddle Tensor
train_data = paddle.to_tensor(X_train.astype('float32'))
train_labels = paddle.to_tensor(y_train)
test_data = paddle.to_tensor(X_test.astype('float32'))
test_labels = paddle.to_tensor(y_test)

# 定义逻辑回归模型
class LogisticRegressionModel(nn.Layer):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.001)

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.clear_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_labels.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# 评估模型 (与logistic_regression_model.py保持一致)
model.eval()
with paddle.no_grad():
    logits = model(test_data)
    y_proba = paddle.nn.functional.sigmoid(logits).numpy().flatten()
    y_pred = (y_proba > 0.5).astype(int)

print("\n评估结果:")
print("准确率:", accuracy_score(y_test, y_pred))
print("召回率:", recall_score(y_test, y_pred))
print("F1分数:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# # 保存模型
# paddle.save(model.state_dict(), 'paddle_lr_model.pdparams')