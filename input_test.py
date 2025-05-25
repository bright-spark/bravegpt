#!/usr/bin/env python3
"""
Input Test Script - Tests different input methods to identify which one works best
in the current environment.
"""

import sys
import os
import time

def test_input_methods():
    """Test different input methods to see which one works reliably"""
    print("\n===== BraveGPT Input Testing Tool =====")
    print("This script will test different input methods to find what works in your environment.")
    print("For each test, try typing some text and pressing Enter.\n")
    
    # Method 1: Standard input()
    print("\n----- Test 1: Standard input() -----")
    print("Type something and press Enter: ", end="")
    try:
        start_time = time.time()
        result = input()
        elapsed = time.time() - start_time
        print(f"Success! Received: '{result}' (took {elapsed:.2f} seconds)")
    except Exception as e:
        print(f"Error with standard input(): {e}")
    
    # Method 2: sys.stdin.readline()
    print("\n----- Test 2: sys.stdin.readline() -----")
    print("Type something and press Enter: ", end="")
    sys.stdout.flush()
    try:
        start_time = time.time()
        result = sys.stdin.readline().strip()
        elapsed = time.time() - start_time
        print(f"Success! Received: '{result}' (took {elapsed:.2f} seconds)")
    except Exception as e:
        print(f"Error with sys.stdin.readline(): {e}")
    
    # Method 3: os.read from stdin
    print("\n----- Test 3: os.read from stdin -----")
    print("Type something and press Enter: ", end="")
    sys.stdout.flush()
    try:
        start_time = time.time()
        result = os.read(0, 1024).decode('utf-8').strip()
        elapsed = time.time() - start_time
        print(f"Success! Received: '{result}' (took {elapsed:.2f} seconds)")
    except Exception as e:
        print(f"Error with os.read: {e}")
    
    print("\n===== Testing Complete =====")
    print("Based on the results, you can determine which input method works best in your environment.")
    print("Use the working method in your BraveGPT application.")

if __name__ == "__main__":
    test_input_methods()
