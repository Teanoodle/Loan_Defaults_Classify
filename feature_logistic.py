import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path='process_data.csv'):
    """加载数据并进行预处理"""
    data = pd.read_csv(file_path)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    return X, y


# ==================================================================================================

def feature_selection(X, y, k=10):
    """特征选择"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features

def create_polynomial_features(X, degree=2):
    """创建多项式特征"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)

# def create_interaction_features(X, selected_features, top_n=3):
#     """创建交互特征"""
#     X_interact = X.copy()
#     top_features = selected_features[:top_n]
#     for i in range(len(top_features)):
#         for j in range(i+1, len(top_features)):
#             col_name = f"{top_features[i]}_x_{top_features[j]}"
#             X_interact[col_name] = X[top_features[i]] * X[top_features[j]]
#     return X_interact

def prepare_features(X, y, k=10, degree=2, top_n=3):
    """完整的特征工程流程"""
    # 特征选择
    X_selected, selected_features = feature_selection(X, y, k=k)
    
    # 输出特征选择结果
    print("\n=== 特征工程详细输出 ===")
    print(f"原始特征数量: {X.shape[1]}")
    print(f"选择的重要特征 ({len(selected_features)}个):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i:2d}. {feat}")
    
    # 多项式特征
    X_poly = create_polynomial_features(X[selected_features], degree=degree)
    print(f"\n生成的多项式特征: {X_poly.shape[1]-len(selected_features)}个")
    
    # 合并特征
    X_engineered = np.hstack([X_selected, X_poly])
    print(f"最终特征总数: {X_engineered.shape[1]}")
    
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n=== 特征工程完成 ===")
    print(f"最终特征维度: {X_engineered.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler

# # 添加直接运行时的功能
# if __name__ == "__main__":
#     print("=== 特征工程独立运行模式 ===")
#     X, y = load_and_preprocess_data()
#     prepare_features(X, y)
#     print("\n提示：这是特征工程的独立运行模式，仅展示特征信息。")
#     print("要训练模型，请运行 logistic_regression_model.py")