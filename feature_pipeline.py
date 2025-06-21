import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

class FeaturePipeline:
    def __init__(self, model_type='random_forest'):
        """
        初始化特征工程管道
        :param model_type: 模型类型，决定使用哪种特征选择方法
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.selector = None
        
        # 根据模型类型初始化特征选择器
        if model_type == 'random_forest':
            self.base_model = RandomForestClassifier()
        elif model_type == 'xgboost':
            self.base_model = XGBClassifier()
        elif model_type == 'lightgbm':
            self.base_model = LGBMClassifier()
        elif model_type == 'logistic':
            self.base_model = LogisticRegression()
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    def fit_transform(self, X, y):
        """
        拟合并转换特征
        :param X: 特征数据
        :param y: 标签数据
        :return: 转换后的特征数据
        """
        # 标准化处理（逻辑回归需要，其他模型可选）
        if self.model_type == 'logistic':
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
            
        # 特征选择
        self.selector = SelectFromModel(self.base_model, threshold='median')
        self.selector.fit(X_scaled, y)
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected

    def transform(self, X):
        """
        应用已拟合的转换
        :param X: 特征数据
        :return: 转换后的特征数据
        """
        if self.selector is None:
            raise RuntimeError("必须先调用fit_transform方法")
            
        # 标准化处理
        if self.model_type == 'logistic':
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
            
        # 特征选择
        X_selected = self.selector.transform(X_scaled)
        
        return X_selected
