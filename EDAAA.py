import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split, cross_validate
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score, make_scorer
# 1. 数据加载
df = pd.read_csv('cleaned_credit_risk_dataset.csv')

# 2. 基础EDA分析
print("=== 数据概览 ===")
print(f"行数: {df.shape[0]}, 列数: {df.shape[1]}")

print("\n=== 数据类型 ===")
print(df.dtypes)

print("\n=== 缺失值统计 ===")
print(df.isnull().sum())

print("\n=== 数值特征描述 ===")
print(df.describe())

# 3. 类别特征分析
cat_cols = df.select_dtypes(include=['object']).columns
print("\n=== 类别特征分布 ===")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

# 4. 目标变量分布
print("\n=== 目标变量分布 ===")
print(df['loan_status'].value_counts(normalize=True))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution')
plt.show()

# 5. 数值特征分布分析 - 重点分析person_income和loan_amnt
target_cols = ['person_income', 'loan_amnt']
for col in target_cols:
    plt.figure(figsize=(12, 6))
    
    # 原始数据分布
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Original {col} Distribution')
    
    # 对数变换后分布
    plt.subplot(1, 2, 2)
    log_transformed = np.log1p(df[col])
    sns.histplot(log_transformed, kde=True)
    plt.title(f'Log Transformed {col} Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # 将变换后的数据添加到DataFrame中
    df[f'log_{col}'] = log_transformed

# 将loan_int_rate转换为百分比格式再转为数值型
if 'loan_int_rate' in df.columns:
    # 先转换为百分比字符串
    df['loan_int_rate'] = df['loan_int_rate'].astype(str) + '%'
    # 再转换为数值型(去掉百分号并除以100)
    df['loan_int_rate'] = df['loan_int_rate'].str.rstrip('%').astype(float) / 100

# 6. 特征相关性分析
plt.figure(figsize=(10,8))
sns.heatmap(df[df.select_dtypes(include=[np.number]).columns].corr(), annot=True, cmap='coolwarm')
plt.title('Numerical Features Correlation')
plt.show()

for col in cat_cols:
    plt.figure(figsize=(10,4))
    sns.countplot(x=col, hue='loan_status', data=df)
    plt.title(f'{col} vs Loan Status')
    plt.xticks(rotation=45)
    plt.show()


    
# 保存修改后的数据到新CSV文件
print("\n=== 正在保存数据 ===")
output_path = os.path.abspath('linear_model.csv')
print(f"保存路径: {output_path}")
print(f"数据行数: {len(df)}")
df.to_csv(output_path, index=False)
print("=== 保存完成 ===")
print(f"文件存在性检查: {os.path.exists(output_path)}")