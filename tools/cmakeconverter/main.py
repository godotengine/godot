#!/usr/bin/env python3

import os
import sys
from converter import convert_build_system

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_sconstruct> [output_path]")
        sys.exit(1)

    sconstruct_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(sconstruct_path):
        print(f"Error: SCons script not found at {sconstruct_path}")
        sys.exit(1)

    # Default output path is CMakeLists.txt in the same directory as SConstruct
    output_path = os.path.join(os.path.dirname(sconstruct_path), "CMakeLists.txt")
    if len(sys.argv) > 2:
        output_path = os.path.abspath(sys.argv[2])

    try:
        convert_build_system(sconstruct_path, output_path)
    except Exception as e:
        import traceback
        print(f"Error during conversion: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()