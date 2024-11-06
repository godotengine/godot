import subprocess


def can_build(env, platform):
    if platform in ("ios", "web", "android"):
        return False

    try:
        subprocess.check_output(["rustup", "--version"], stderr=subprocess.STDOUT)
        subprocess.check_output(["cargo", "--version"], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("Cargo or Rustup not found. mvsqlite build skipped.")
        if platform == "windows":
            print("Use `scoop install rustup-gnu` to install rust.")
        return False

    if platform == "windows":
        use_mingw = env["use_mingw"]
        if not use_mingw:
            print("MSVC target is installed. mvsqlite build skipped.")
            return False

        try:
            subprocess.check_output(["rustup", "target", "list", "--installed"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print("Windows Rust target not found. Installing...")
            subprocess.check_call(["rustup", "target", "add", "x86_64-pc-windows-gnu"])  # or other Windows target
    else:
        try:
            subprocess.check_output(["rustup", "target", "list", "--installed"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print("x86_64-apple-darwin target not found. Installing...")
            subprocess.check_call(["rustup", "target", "add", "x86_64-apple-darwin"])

        try:
            subprocess.check_output(["rustup", "target", "list", "--installed"], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print("aarch64-apple-darwin target not found. Installing...")
            subprocess.check_call(["rustup", "target", "add", "aarch64-apple-darwin"])

    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "SQLite",
        "SQLiteQuery",
    ]


def get_doc_path():
    return "doc_classes"
