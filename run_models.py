import subprocess
import sys

def run_script(script_name):
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"\n=== {script_name} 输出 ===")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n=== {script_name} 执行出错 ===")
        print(e.stderr)

if __name__ == "__main__":
    print("开始运行所有模型...")
    
    models = [
        "logistic_regression.py",
        "xgboost_model.py", 
        "lightgbm_model.py",
        "random_forest.py",
        "ensemble_model.py"
    ]
    
    for model in models:
        run_script(model)
        
    print("\n所有模型运行完成！")