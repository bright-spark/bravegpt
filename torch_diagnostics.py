#!/usr/bin/env python3
"""
PyTorch Diagnostics Tool - Tests PyTorch installation and compatibility
"""

import sys
import os
import platform
import subprocess

def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")

def run_command(cmd):
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, 
                               capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e}\n{e.stderr}"

def check_system_info():
    """Check system information"""
    print_section("System Information")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"In virtual environment: {in_venv}")
    print(f"Python executable: {sys.executable}")

def check_pytorch_installation():
    """Check PyTorch installation"""
    print_section("PyTorch Installation")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch path: {torch.__file__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check device info
        print("\nDevice Information:")
        print(f"Default device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        
        # Check if get_default_device exists
        has_get_default_device = hasattr(torch, 'get_default_device')
        print(f"\nHas torch.get_default_device(): {has_get_default_device}")
        
        # Check for _C._get_default_device
        has_c_get_default_device = hasattr(torch._C, '_get_default_device')
        print(f"Has torch._C._get_default_device(): {has_c_get_default_device}")
        
        # Try to use device methods
        print("\nTesting device creation:")
        print(f"torch.device('cpu'): {torch.device('cpu')}")
        if torch.cuda.is_available():
            print(f"torch.device('cuda'): {torch.device('cuda')}")
        
        # Test tensor creation
        print("\nTesting tensor creation:")
        cpu_tensor = torch.tensor([1, 2, 3])
        print(f"CPU tensor: {cpu_tensor}, device: {cpu_tensor.device}")
        
    except ImportError:
        print("PyTorch is not installed.")
    except Exception as e:
        print(f"Error checking PyTorch: {e}")

def check_transformers_installation():
    """Check Transformers installation"""
    print_section("Transformers Installation")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        print(f"Transformers path: {transformers.__file__}")
        
        # Check for the problematic function
        import inspect
        from transformers import modeling_utils
        
        print("\nChecking for get_default_device usage in transformers:")
        
        # Look for get_default_device in modeling_utils.py
        with open(modeling_utils.__file__, 'r') as f:
            content = f.read()
            if 'get_default_device' in content:
                print("Found 'get_default_device' in modeling_utils.py")
                
                # Find the specific function using it
                for name, func in inspect.getmembers(modeling_utils, inspect.isfunction):
                    source = inspect.getsource(func)
                    if 'get_default_device' in source:
                        print(f"  - Used in function: {name}")
            else:
                print("No 'get_default_device' usage found in modeling_utils.py")
        
    except ImportError:
        print("Transformers is not installed.")
    except Exception as e:
        print(f"Error checking Transformers: {e}")

def check_installed_packages():
    """Check installed packages"""
    print_section("Installed Packages")
    
    pip_list = run_command(f"{sys.executable} -m pip list")
    
    # Filter for relevant packages
    relevant_packages = []
    for line in pip_list.split('\n'):
        if any(pkg in line.lower() for pkg in ['torch', 'transformers', 'safetensors', 'accelerate']):
            relevant_packages.append(line)
    
    if relevant_packages:
        print("Relevant installed packages:")
        for pkg in relevant_packages:
            print(f"  {pkg}")
    else:
        print("No relevant packages found.")

def main():
    """Main function to run diagnostics"""
    print("\n📊 PyTorch & Transformers Diagnostics Tool 📊")
    print("This tool will help diagnose issues with PyTorch and Transformers.")
    
    check_system_info()
    check_pytorch_installation()
    check_transformers_installation()
    check_installed_packages()
    
    print_section("Recommendations")
    print("Based on the diagnostics:")
    print("1. If PyTorch version is >= 2.0.0 and transformers is >= 4.30.0,")
    print("   you may need to downgrade transformers to a compatible version.")
    print("2. Try: pip install transformers<4.30.0")
    print("3. If using a custom model, ensure it's compatible with your PyTorch version.")
    print("4. For input handling issues, use sys.stdin.readline() instead of input().")
    
    print("\nDiagnostics complete! Use this information to fix your BraveGPT issues.")

if __name__ == "__main__":
    main()
