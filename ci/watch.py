#!/usr/bin/env python3
"""
GodotGS File Watcher
====================

Monitors the Gaussian Splatting module for changes and automatically triggers CI pipeline.
"""

import time
import sys
from pathlib import Path
from typing import Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import colorama
    from colorama import Fore, Style
    colorama.init()
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "ci/requirements.txt"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import colorama
    from colorama import Fore, Style
    colorama.init()

# Import our pipeline
import simple_pipeline


class GodotGSChangeHandler(FileSystemEventHandler):
    """Handles file system changes for the Gaussian Splatting module"""

    def __init__(self, project_root, debounce_seconds: float = 5.0):
        self.project_root = project_root
        self.debounce_seconds = debounce_seconds
        self.last_run = 0
        self.pending_changes: Set[str] = set()

        # File extensions to watch
        self.watched_extensions = {'.cpp', '.h', '.glsl', '.py', '.gd', '.SCsub'}

        # Directories to ignore
        self.ignored_dirs = {'bin', 'Binaries', 'Intermediate', '.git', '__pycache__', 'reports'}

    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return

        self._handle_file_change(event.src_path, "modified")

    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return

        self._handle_file_change(event.src_path, "created")

    def on_deleted(self, event):
        """Handle file deletion events"""
        if event.is_directory:
            return

        self._handle_file_change(event.src_path, "deleted")

    def _handle_file_change(self, file_path: str, change_type: str):
        """Process a file change event"""
        path = Path(file_path)

        # Check if we should ignore this change
        if self._should_ignore_change(path):
            return

        self.pending_changes.add(f"{change_type}: {path.name}")

        print(f"{Fore.YELLOW}📝 Detected {change_type}: {path.relative_to(Path.cwd())}{Style.RESET_ALL}")

        # Debounce: Only run if enough time has passed since last run
        current_time = time.time()
        if current_time - self.last_run > self.debounce_seconds:
            self._trigger_pipeline()

    def _should_ignore_change(self, path: Path) -> bool:
        """Check if we should ignore this file change"""
        # Check file extension
        if path.suffix not in self.watched_extensions:
            return True

        # Check if in ignored directory
        for part in path.parts:
            if part in self.ignored_dirs:
                return True

        # Check if it's a temporary file
        if path.name.startswith('.') or path.name.endswith('.tmp'):
            return True

        return False

    def _trigger_pipeline(self):
        """Trigger the CI pipeline"""
        self.last_run = time.time()

        print(f"\n{Fore.CYAN}🔄 TRIGGERING CI PIPELINE{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Changes detected:{Style.RESET_ALL}")
        for change in sorted(self.pending_changes):
            print(f"  • {change}")

        self.pending_changes.clear()

        print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")

        try:
            # Run the simple pipeline
            result = simple_pipeline.main()
            if result == 0:
                print(f"{Fore.GREEN}Pipeline completed successfully!{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Pipeline failed!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Pipeline error: {e}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}👀 Continuing to watch for changes...{Style.RESET_ALL}\n")


def main():
    """Main entry point for file watcher"""
    project_root = Path(__file__).parent.parent

    print(f"{Fore.CYAN}👀 GodotGS File Watcher Starting{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    print(f"Project root: {project_root}")
    print("Watching: modules/gaussian_splatting/")
    print(f"Debounce: 5 seconds")
    print(f"Extensions: .cpp, .h, .glsl, .py, .gd, .SCsub")
    print()

    # Run pipeline once at startup
    print(f"{Fore.YELLOW}Running initial pipeline...{Style.RESET_ALL}")
    try:
        # Temporarily set sys.argv to run without simulation
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]  # Keep only script name

        result = simple_pipeline.main()
        sys.argv = original_argv  # Restore original argv

        if result == 0:
            print(f"{Fore.GREEN}Initial pipeline completed successfully{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}Initial pipeline had issues (continuing to watch){Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.YELLOW}Initial pipeline failed: {e} (continuing to watch){Style.RESET_ALL}")

    print()

    # Setup file watching
    observer = Observer()
    handler = GodotGSChangeHandler(project_root)

    # Watch the module directory
    module_path = project_root / "modules" / "gaussian_splatting"
    if module_path.exists():
        observer.schedule(handler, str(module_path), recursive=True)
        print(f"{Fore.GREEN}📁 Watching: {module_path}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}❌ Module path not found: {module_path}{Style.RESET_ALL}")
        return 1

    # Also watch build scripts and CI scripts
    ci_path = project_root / "ci"
    if ci_path.exists():
        observer.schedule(handler, str(ci_path), recursive=True)
        print(f"{Fore.GREEN}📁 Also watching: {ci_path}{Style.RESET_ALL}")

    observer.start()

    print(f"\n{Fore.GREEN}👀 File watcher started! Press Ctrl+C to stop.{Style.RESET_ALL}\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}🛑 Stopping file watcher...{Style.RESET_ALL}")
        observer.stop()

    observer.join()
    print(f"{Fore.GREEN}✅ File watcher stopped.{Style.RESET_ALL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
