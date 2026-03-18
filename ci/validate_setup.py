#!/usr/bin/env python3
"""
GodotGS Setup Validation
=======================

Validates that the development environment is properly configured for CI/CD.
"""

import subprocess
import sys
from pathlib import Path

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
except ImportError:
    class Fore:
        GREEN = YELLOW = RED = CYAN = ""
    class Style:
        RESET_ALL = ""


def check_python():
    """Check Python version"""
    print(f"{Fore.YELLOW}Checking Python...{Style.RESET_ALL}")

    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  {Fore.GREEN}Python {version.major}.{version.minor}.{version.micro} - OK{Style.RESET_ALL}")
        return True
    else:
        print(f"  {Fore.RED}Python {version.major}.{version.minor}.{version.micro} - Need 3.8+{Style.RESET_ALL}")
        return False


def check_scons():
    """Check if SCons is available"""
    print(f"{Fore.YELLOW}Checking SCons...{Style.RESET_ALL}")

    try:
        result = subprocess.run(["scons", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  {Fore.GREEN}{version_line} - OK{Style.RESET_ALL}")
            return True
        else:
            print(f"  {Fore.RED}SCons found but not working{Style.RESET_ALL}")
            return False
    except FileNotFoundError:
        print(f"  {Fore.RED}SCons not found{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Install with: pip install scons{Style.RESET_ALL}")
        return False


def check_godot_source():
    """Check if the repo root contains a buildable Godot tree."""
    print(f"{Fore.YELLOW}Checking engine root...{Style.RESET_ALL}")

    project_root = Path(__file__).parent.parent
    godot_source = project_root

    if godot_source.exists():
        scons_file = godot_source / "SConstruct"
        if scons_file.exists():
            print(f"  {Fore.GREEN}Engine root found at {godot_source} - OK{Style.RESET_ALL}")
            return True
        else:
            print(f"  {Fore.RED}Engine root is missing SConstruct{Style.RESET_ALL}")
            return False
    else:
        print(f"  {Fore.RED}Engine root not found at {godot_source}{Style.RESET_ALL}")
        return False


def check_module():
    """Check if Gaussian Splatting module is available"""
    print(f"{Fore.YELLOW}Checking Gaussian Splatting module...{Style.RESET_ALL}")

    project_root = Path(__file__).parent.parent
    module_path = project_root / "modules" / "gaussian_splatting"

    if module_path.exists():
        config_file = module_path / "config.py"
        register_file = module_path / "register_types.h"

        if config_file.exists() and register_file.exists():
            print(f"  {Fore.GREEN}Module found at {module_path} - OK{Style.RESET_ALL}")
            return True
        else:
            print(f"  {Fore.RED}Module missing required files{Style.RESET_ALL}")
            return False
    else:
        print(f"  {Fore.RED}Module not found at {module_path}{Style.RESET_ALL}")
        return False


def check_existing_binary():
    """Check if Godot binary already exists"""
    print(f"{Fore.YELLOW}Checking for existing Godot binary...{Style.RESET_ALL}")

    project_root = Path(__file__).parent.parent
    godot_source = project_root
    binary_path = godot_source / "bin" / "godot.windows.editor.x86_64.exe"

    if binary_path.exists():
        print(f"  {Fore.GREEN}Binary found at {binary_path} - OK{Style.RESET_ALL}")

        # Try to run it quickly to see if it works
        try:
            result = subprocess.run([str(binary_path), "--version", "--quiet"],
                                   capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"  {Fore.GREEN}Binary is functional - OK{Style.RESET_ALL}")
                return True
            else:
                print(f"  {Fore.YELLOW}Binary exists but may have issues{Style.RESET_ALL}")
                return False
        except Exception as e:
            print(f"  {Fore.YELLOW}Binary exists but cannot test: {e}{Style.RESET_ALL}")
            return False
    else:
        print(f"  {Fore.YELLOW}No existing binary found{Style.RESET_ALL}")
        return False


def main():
    """Main validation function"""
    print(f"\n{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}GodotGS Development Environment Validation{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

    checks = [
        ("Python", check_python),
        ("SCons", check_scons),
        ("Engine Root", check_godot_source),
        ("Module", check_module),
        ("Existing Binary", check_existing_binary),
    ]

    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
        print()

    # Summary
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Summary:{Style.RESET_ALL}")

    all_passed = True
    critical_failed = False

    for name, passed in results:
        status = f"{Fore.GREEN}PASS" if passed else f"{Fore.RED}FAIL"
        print(f"  {name:<20} {status}{Style.RESET_ALL}")

        if not passed:
            all_passed = False
            if name in ["Python", "Engine Root", "Module"]:
                critical_failed = True

    print()

    if all_passed:
        print(f"{Fore.GREEN}All checks passed! CI/CD pipeline ready to run.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Run: python ci/simple_pipeline.py{Style.RESET_ALL}")
        return 0
    elif critical_failed:
        print(f"{Fore.RED}Critical components missing. Please fix before running CI/CD.{Style.RESET_ALL}")
        return 1
    else:
        print(f"{Fore.YELLOW}Some optional components missing, but CI/CD can run in simulation mode.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Run: python ci/simple_pipeline.py --simulate{Style.RESET_ALL}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
