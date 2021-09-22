import subprocess
import sys
import os
import pathlib

ninja_builddir = pathlib.Path(sys.argv[2])

if not os.path.exists(ninja_builddir / "scons_daemon_dirty"):
    cmd = [sys.executable, str(pathlib.Path(__file__).parent / "ninja_scons_daemon.py")] + sys.argv[1:]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
    with open(ninja_builddir / "scons_daemon_dirty", "w") as f:
        f.write(str(p.pid))
