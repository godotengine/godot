#!/usr/bin/env python3
"""
TEST RUNNER FOR GODOT OPTIMIZATIONS
Run real-world examples and benchmarks
"""

import os
import sys
import subprocess
import time

GODOT_BIN = "bin/godot.windows.editor.dev.x86_64.console.exe"

EXAMPLES = [
    {
        "name": "Real-World Examples",
        "script": "REAL_WORLD_EXAMPLES.gd",
        "description": "All 5 optimizations demonstrated with real game scenarios"
    },
    {
        "name": "Struct Performance Benchmark",
        "script": "benchmarks/struct_performance.gd",
        "description": "Struct vs Dict vs Class performance comparison"
    },
    {
        "name": "Array Reserve Test",
        "script": "tests/array_reserve_test.gd",
        "description": "Tests Array.reserve() optimization (#2)"
    },
    {
        "name": "Typed Iteration Test",
        "script": "tests/typed_iteration_test.gd",
        "description": "Tests OPCODE_ITERATE_TYPED_ARRAY (#1)"
    },
    {
        "name": "Dead Code Elimination Test",
        "script": "tests/dead_code_test.gd",
        "description": "Tests compile-time constant folding (#6)"
    }
]

def check_godot_binary():
    """Check if Godot binary exists"""
    if not os.path.exists(GODOT_BIN):
        print(f"❌ Godot binary not found: {GODOT_BIN}")
        print("   Run: python -m SCons platform=windows target=editor dev_build=yes")
        return False
    print(f"✅ Found Godot binary: {GODOT_BIN}")
    return True

def run_example(example):
    """Run a single example script"""
    script_path = example["script"]
    
    if not os.path.exists(script_path):
        print(f"⚠️  Script not found: {script_path}")
        return False
    
    print(f"\n{'='*80}")
    print(f"🚀 Running: {example['name']}")
    print(f"   {example['description']}")
    print(f"   Script: {script_path}")
    print(f"{'='*80}\n")
    
    try:
        # Run Godot with the script
        cmd = [GODOT_BIN, "--headless", "--script", script_path]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ Example failed with exit code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
        else:
            print(f"\n✅ Example completed in {elapsed:.2f}s")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ Example timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        return False

def main():
    """Main entry point"""
    print("="*80)
    print("🎮 GODOT OPTIMIZATION TEST RUNNER")
    print("="*80)
    
    # Check Godot binary
    if not check_godot_binary():
        sys.exit(1)
    
    print(f"\nFound {len(EXAMPLES)} examples to run")
    
    # Run all examples
    results = []
    for i, example in enumerate(EXAMPLES, 1):
        print(f"\n[{i}/{len(EXAMPLES)}]")
        success = run_example(example)
        results.append((example["name"], success))
        
        if i < len(EXAMPLES):
            print("\nWaiting 2 seconds before next example...")
            time.sleep(2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print("="*80)
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
