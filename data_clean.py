import pandas as pd
import numpy as np

def clean_data(input_file, output_file, categorical_encoding='onehot'):
    """
    Clean up the data and handle categorical variables
    
    parameters:
        input_file: Enter the path of the CSV file
        output_file: Output the path of the CSV file
        categorical_encoding: The encoding method of categorical variables:
        
                                     "onehot" - Exclusive hot Encoding (default)
                                     'label' - Label code
                                     "frequency" - Frequency coding
    """
    # Read data
    df = pd.read_csv(input_file)

    # Identify numerical columns and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Save the original classification column names for subsequent processing
    original_categorical_cols = categorical_cols.copy()
    
    # Fill in the numerical missing values with the median
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
    
    # Fill in the missing values of categorical variables with the mode
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    
    # Encoding processing of categorical variables
    if categorical_encoding == 'onehot':
        # One hot encoding - Ensure that all categorical variables are converted to 0/1 format
        df = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_', dtype=int)
        
        # Process the cb_person_default_on_file column and convert Y/N to 1/0
        if 'cb_person_default_on_file_Y' in df.columns:
            df['cb_person_default_on_file'] = df['cb_person_default_on_file_Y'].astype(int)
            df.drop(['cb_person_default_on_file_N', 'cb_person_default_on_file_Y'], axis=1, inplace=True)


    elif categorical_encoding == 'label':
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])


    elif categorical_encoding == 'frequency':
        # Frequency encoding
        for col in categorical_cols:
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq)
    else:
        raise ValueError(f"Unsupported encoding methods: {categorical_encoding}")
    
    # Dealing with outliers and abnormal records.
    # Delete abnormal records whose age is over 80 years old
    print(f"\n The number of rows of the original data: {len(df)}")
    df = df[df['person_age'] <= 80]
    print(f"The number of lines after deleting records where the age is over 80 years old: {len(df)}")
    
    # Delete records with an income greater than 1M
    df = df[df['person_income'] <= 1000000]
    print(f"The number of lines after deleting the record where the income is greater than 1M: {len(df)}")
    
    # Delete the records where the years of employment are longer than the age
    df = df[df['person_emp_length'] <= df['person_age']]
    print(f"The number of lines after deleting the record of employment whose years > age: {len(df)}")
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    
    # Check if there are still missing values
    missing_values = df.isnull().sum().sum()
    print(f" \n Missing value check: There are a total of {missing_values} missing values in the dataset")
    rows, cols = df.shape
    print(f"The data consists of {rows} rows and {cols} columns")
    
    # Numerical variable descriptions
    numeric_desc = df.describe()
    
    # Categorical variable descriptions
    categorical_desc = {}
    if categorical_encoding == 'onehot':
        # For the one hot encoding, the original data statistics before encoding are used
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
        # For other encoding methods, use the encoded data for statistics
        for col in original_categorical_cols:
            freq = df[col].value_counts(normalize=True)
            categorical_desc[col] = {
                'unique_count': df[col].nunique(),
                'top_category': freq.idxmax(),
                'top_freq': freq.max(),
                'freq_distribution': freq.to_dict()
            }
    
    # Return the merged descriptive statistics
    return {
        'numeric_variables': numeric_desc,
        'categorical_variables': categorical_desc,
        'encoding_method': categorical_encoding,
        'original_columns': list(original_categorical_cols)
    }

if __name__ == "__main__":
    input_csv = "cleaned_credit_risk_dataset.csv"
    output_csv = "cleaned_credit_risk_dataset_processed.csv"
    
    # The example uses different encoding methods
    print("\n Use One Hot encoding to process categorical variables:")
    results = clean_data(input_csv, output_csv, categorical_encoding='onehot')
    print("The data cleaning has been completed. Descriptive statistics: ")
    
    print("\n Numeric variables statistics:")
    print(results['numeric_variables'])
    
    print("\n Categorical variables statistics:")
    for var, stats in results['categorical_variables'].items():
        print(f"\n{var}:")
        print(f"  Unique count: {stats['unique_count']}")
        print(f"  top category: {stats['top_category']} (Proportion: {stats['top_freq']:.2%})")
    
