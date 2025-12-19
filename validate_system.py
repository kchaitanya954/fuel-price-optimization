"""
Validation script to check system structure and imports.
"""
import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are available."""
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'fastapi',
        'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    return missing


def check_files():
    """Check if all required files exist."""
    required_files = [
        'config.py',
        'data_pipeline.py',
        'feature_engineering.py',
        'model.py',
        'price_recommender.py',
        'api.py',
        'explore_data.py',
        'oil_retail_history.csv',
        'today_example.json',
        'requirements.txt',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - NOT FOUND")
            missing.append(file)
    
    return missing


def main():
    """Main validation function."""
    print("="*60)
    print("SYSTEM VALIDATION")
    print("="*60)
    
    print("\n1. Checking required packages...")
    missing_packages = check_imports()
    
    print("\n2. Checking required files...")
    missing_files = check_files()
    
    print("\n" + "="*60)
    if missing_packages:
        print(f"WARNING: {len(missing_packages)} package(s) missing.")
        print("Install with: pip install -r requirements.txt")
    else:
        print("All packages are installed.")
    
    if missing_files:
        print(f"WARNING: {len(missing_files)} file(s) missing.")
    else:
        print("All required files are present.")
    
    if not missing_packages and not missing_files:
        print("\n✓ System is ready to use!")
    else:
        print("\n⚠ Please resolve the issues above before using the system.")
    
    print("="*60)


if __name__ == "__main__":
    main()

