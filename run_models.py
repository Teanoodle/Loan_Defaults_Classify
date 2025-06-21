import subprocess
import sys

def run_script(script_name):
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"\n=== {script_name} output ===")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n=== {script_name} execution error ===")
        print(e.stderr)

if __name__ == "__main__":
    print("Starting to run all models...")
    
    models = [
        "logistic_regression.py",
        "xgboost_model.py", 
        "lightgbm_model.py",
        "random_forest.py",
        "ensemble_model.py"
    ]
    
    for model in models:
        run_script(model)
        
    print("\nAll models completed running!")