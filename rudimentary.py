import os
import pandas as pd

def analyze_indian_files():
    """Analyze only Indian car data files (Indian.csv and Indian_v2.csv)"""
    # Define target files
    target_files = ['Indian.csv', 'Indian_v2.csv']
    
    # Check if files exist
    existing_files = [f for f in target_files if os.path.exists(f)]
    
    if not existing_files:
        print("No Indian data files found in current directory")
        return
    
    print(f"Found {len(existing_files)} Indian data file(s):")
    
    for file in existing_files:
        print(f"\nAnalyzing file: {file}")
        try:
            # Read CSV file
            df = pd.read_csv(file)
            
            # Basic analysis
            print(f"Total rows: {len(df)}")
            print(f"Total columns: {len(df.columns)}")
            print("\nColumn names and data types:")
            print(df.dtypes)
            
            print("\nMissing values count:")
            print(df.isnull().sum())
            
            print("\nFirst 5 rows preview:")
            print(df.head())
            
        except Exception as e:
            print(f"Error analyzing {file}: {str(e)}")

if __name__ == "__main__":
    try:
        analyze_indian_files()
    except ImportError:
        print("Required dependencies missing. Please install pandas first:")
        print("pip install pandas")
