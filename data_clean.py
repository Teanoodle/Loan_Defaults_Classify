import pandas as pd
import numpy as np

def clean_data(input_file, output_file, categorical_encoding='onehot'):
    """
    清理数据并处理分类变量
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        categorical_encoding: 分类变量编码方式
            'onehot' - 独热编码(默认)
            'label' - 标签编码
            'frequency' - 频率编码
    """
    # 读取数据
    df = pd.read_csv(input_file)

    # 识别数值型列和分类列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # 保存原始分类列名用于后续处理
    original_categorical_cols = categorical_cols.copy()
    
    # 用中位数填充数值型缺失值
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    
    # 用众数填充分类变量缺失值
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    
    # 分类变量编码处理
    if categorical_encoding == 'onehot':
        # 独热编码 - 确保所有分类变量都转换为0/1格式
        df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_', dtype=int)
        
        # 处理cb_person_default_on_file列，将Y/N转换为1/0
        if 'cb_person_default_on_file_Y' in df.columns:
            df['cb_person_default_on_file'] = df['cb_person_default_on_file_Y'].astype(int)
            df.drop(['cb_person_default_on_file_N', 'cb_person_default_on_file_Y'], axis=1, inplace=True)
    elif categorical_encoding == 'label':
        # 标签编码
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    elif categorical_encoding == 'frequency':
        # 频率编码
        for col in categorical_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
    else:
        raise ValueError(f"不支持的编码方式: {categorical_encoding}")
    
    # 删除年龄大于80岁的异常记录
    print(f"\n原始数据行数: {len(df)}")
    df = df[df['person_age'] <= 80]
    print(f"删除年龄>80岁记录后行数: {len(df)}")
    
    # 删除收入大于1M的记录
    df = df[df['person_income'] <= 1000000]
    print(f"删除收入>1M记录后行数: {len(df)}")
    
    # 删除就业年限大于年龄的记录
    df = df[df['person_emp_length'] <= df['person_age']]
    print(f"删除就业年限>年龄记录后行数: {len(df)}")
    
    # 保存清理后的数据
    df.to_csv(output_file, index=False)
    
    # 检查是否还有缺失值
    missing_values = df.isnull().sum().sum()
    print(f"\n缺失值检查：清理后数据集中共存在 {missing_values} 个缺失值")
    rows, cols = df.shape
    print(f"数据共有 {rows} 行 {cols} 列")
    
    # 生成数值型变量描述
    numeric_desc = df.describe()
    
    # 生成分类变量描述
    categorical_desc = {}
    if categorical_encoding == 'onehot':
        # 对于独热编码，使用编码前的原始数据统计
        original_df = pd.read_csv(input_file)
        for col in original_categorical_cols:
            freq = original_df[col].value_counts(normalize=True)
            categorical_desc[col] = {
                'unique_count': original_df[col].nunique(),
                'top_category': freq.idxmax(),
                'top_freq': freq.max(),
                'freq_distribution': freq.to_dict()
            }
    else:
        # 对于其他编码方式，使用编码后的数据统计
        for col in original_categorical_cols:
            freq = df[col].value_counts(normalize=True)
            categorical_desc[col] = {
                'unique_count': df[col].nunique(),
                'top_category': freq.idxmax(),
                'top_freq': freq.max(),
                'freq_distribution': freq.to_dict()
            }
    
    # 返回合并的描述统计
    return {
        'numeric_variables': numeric_desc,
        'categorical_variables': categorical_desc,
        'encoding_method': categorical_encoding,
        'original_columns': list(original_categorical_cols)
    }

if __name__ == "__main__":
    input_csv = "cleaned_credit_risk_dataset.csv"
    output_csv = "cleaned_credit_risk_dataset_processed.csv"
    
    # 示例使用不同的编码方式
    print("\n使用独热编码处理分类变量:")
    results = clean_data(input_csv, output_csv, categorical_encoding='onehot')
    print("数据清理完成，描述统计如下：")
    
    print("\n数值变量统计:")
    print(results['numeric_variables'])
    
    print("\n分类变量统计:")
    for var, stats in results['categorical_variables'].items():
        print(f"\n{var}:")
        print(f"  唯一值数量: {stats['unique_count']}")
        print(f"  最常见类别: {stats['top_category']} (占比: {stats['top_freq']:.2%})")
    
