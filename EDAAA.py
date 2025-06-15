import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Data loading
df = pd.read_csv('cleaned_data.csv')

# 2. Basic EDA Analysis
print("=== Data Overview ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== Data Type ===")
print(df.dtypes)

print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== Data description ===")
print(df.describe())

# 3. Categorical feature
cat_cols = df.select_dtypes(include=['object']).columns
print("\n=== Categorical feature distribution ===")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

# 4. Target variable
print("\n=== Target variable distribution ===")
print(df['loan_status'].value_counts(normalize=True))
sns.countplot(x='loan_status', data=df)
plt.title('Loan Status Distribution')
plt.show()

# 5. Log transfermation on person_income and loan_amnt
target_cols = ['person_income', 'loan_amnt']
for col in target_cols:
    plt.figure(figsize=(12, 6))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Original {col} Distribution')
    
    # Log transformed distribution
    plt.subplot(1, 2, 2)
    log_transformed = np.log1p(df[col])
    sns.histplot(log_transformed, kde=True)
    plt.title(f'Log Transformed {col} Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # add log transformed column to DataFrame
    df[f'log_{col}'] = log_transformed

# transform 'loan_int_rate' to numeric
if 'loan_int_rate' in df.columns:
    # first convert to string and append '%'
    df['loan_int_rate'] = df['loan_int_rate'].astype(str) + '%'
    # then remove '%' and convert to float
    df['loan_int_rate'] = df['loan_int_rate'].str.rstrip('%').astype(float) / 100

# 6. Correlation analysis
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


    
# save the processed DataFrame
print("\n=== Data is being saved ===")
output_path = os.path.abspath('process_data.csv')
print(f"save path: {output_path}")
print(f"rows: {len(df)}")
df.to_csv(output_path, index=False)
print("=== Save complete ===")
print(f"Existence check: {os.path.exists(output_path)}")