import shutil
import subprocess
import sys
from pathlib import Path

M = Path(__file__).resolve().parent

# runs the generators
subprocess.run([sys.executable, "gen_cpp.py", M / "extension_api.json", M / "Godot-Wasm-Exports"], cwd=M, check=True)
subprocess.run([sys.executable, "gen_cs.py", M / "extension_api.json", M / "NET"], cwd=M, check=True)
subprocess.run([sys.executable, "gen_js.py", M / "extension_api.json", M / "Bridge_Functions"], cwd=M, check=True)

# builds dotnet dll
try:
    subprocess.run(["dotnet", "build", M / "NET/GodotWeb.csproj", "-c", "Release"], check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("dotnet not found, skipping C# build")

# copies the contents of Web into. web_zip
src = M / "Web"
dst = M.parents[2] / "bin" / ".web_zip" / "JSMarshals"
if src.exists():
    shutil.copytree(src, dst, dirs_exist_ok=True)
