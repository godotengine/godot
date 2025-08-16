import os
import sys

import common_compiler_flags
import my_spawn
from SCons.Tool import mingw, msvc
from SCons.Variables import BoolVariable


def silence_msvc(env):
    import os
    import re
    import tempfile

    # Ensure we have a location to write captured output to, in case of false positives.
    capture_path = os.path.join(os.path.dirname(__file__), "..", "msvc_capture.log")
    with open(capture_path, "wt", encoding="utf-8"):
        pass

    old_spawn = env["SPAWN"]
    re_redirect_stream = re.compile(r"^[12]?>")
    re_cl_capture = re.compile(r"^.+\.(c|cc|cpp|cxx|c[+]{2})$", re.IGNORECASE)
    re_link_capture = re.compile(r'\s{3}\S.+\s(?:"[^"]+.lib"|\S+.lib)\s.+\s(?:"[^"]+.exp"|\S+.exp)')

    def spawn_capture(sh, escape, cmd, args, env):
        # We only care about cl/link, process everything else as normal.
        if args[0] not in ["cl", "link"]:
            return old_spawn(sh, escape, cmd, args, env)

        # Process as normal if the user is manually rerouting output.
        for arg in args:
            if re_redirect_stream.match(arg):
                return old_spawn(sh, escape, cmd, args, env)

        tmp_stdout, tmp_stdout_name = tempfile.mkstemp()
        os.close(tmp_stdout)
        args.append(f">{tmp_stdout_name}")
        ret = old_spawn(sh, escape, cmd, args, env)

        try:
            with open(tmp_stdout_name, "r", encoding=sys.stdout.encoding, errors="replace") as tmp_stdout:
                lines = tmp_stdout.read().splitlines()
            os.remove(tmp_stdout_name)
        except OSError:
            pass

        # Early process no lines (OSError)
        if not lines:
            return ret

        is_cl = args[0] == "cl"
        content = ""
        caught = False
        for line in lines:
            # These conditions are far from all-encompassing, but are specialized
            # for what can be reasonably expected to show up in the repository.
            if not caught and (is_cl and re_cl_capture.match(line)) or (not is_cl and re_link_capture.match(line)):
                caught = True
                try:
                    with open(capture_path, "a", encoding=sys.stdout.encoding) as log:
                        log.write(line + "\n")
                except OSError:
                    print(f'WARNING: Failed to log captured line: "{line}".')
                continue
            content += line + "\n"
        # Content remaining assumed to be an error/warning.
        if content:
            sys.stderr.write(content)

        return ret

    env["SPAWN"] = spawn_capture


def options(opts):
    mingw = os.getenv("MINGW_PREFIX", "")

    opts.Add(BoolVariable("use_mingw", "Use the MinGW compiler instead of MSVC - only effective on Windows", False))
    opts.Add(BoolVariable("use_static_cpp", "Link MinGW/MSVC C++ runtime libraries statically", True))
    opts.Add(BoolVariable("silence_msvc", "Silence MSVC's cl/link stdout bloat, redirecting errors to stderr.", True))
    opts.Add(BoolVariable("debug_crt", "Compile with MSVC's debug CRT (/MDd)", False))
    opts.Add(BoolVariable("use_llvm", "Use the LLVM compiler (MVSC or MinGW depending on the use_mingw flag)", False))
    opts.Add("mingw_prefix", "MinGW prefix", mingw)


def exists(env):
    return True


def generate(env):
    if not env["use_mingw"] and msvc.exists(env):
        if env["arch"] == "x86_64":
            env["TARGET_ARCH"] = "amd64"
        elif env["arch"] == "arm64":
            env["TARGET_ARCH"] = "arm64"
        elif env["arch"] == "arm32":
            env["TARGET_ARCH"] = "arm"
        elif env["arch"] == "x86_32":
            env["TARGET_ARCH"] = "x86"

        env["MSVC_SETUP_RUN"] = False  # Need to set this to re-run the tool
        env["MSVS_VERSION"] = None
        env["MSVC_VERSION"] = None

        env["is_msvc"] = True

        # MSVC, linker, and archiver.
        msvc.generate(env)
        env.Tool("msvc")
        env.Tool("mslib")
        env.Tool("mslink")

        env.Append(CPPDEFINES=["TYPED_METHOD_BIND", "NOMINMAX"])
        env.Append(CCFLAGS=["/utf-8"])
        env.Append(LINKFLAGS=["/WX"])

        if env["use_llvm"]:
            env["CC"] = "clang-cl"
            env["CXX"] = "clang-cl"

        if env["debug_crt"]:
            # Always use dynamic runtime, static debug CRT breaks thread_local.
            env.AppendUnique(CCFLAGS=["/MDd"])
        else:
            if env["use_static_cpp"]:
                env.AppendUnique(CCFLAGS=["/MT"])
            else:
                env.AppendUnique(CCFLAGS=["/MD"])

        if env["silence_msvc"] and not env.GetOption("clean"):
            silence_msvc(env)

        if not env["use_llvm"]:
            env.AppendUnique(CCFLAGS=["/experimental:external", "/external:anglebrackets"])
        env.AppendUnique(CCFLAGS=["/external:W0"])
        env["EXTINCPREFIX"] = "/external:I"

    elif (sys.platform == "win32" or sys.platform == "msys") and not env["mingw_prefix"]:
        env["use_mingw"] = True
        mingw.generate(env)
        # Don't want lib prefixes
        env["IMPLIBPREFIX"] = ""
        env["SHLIBPREFIX"] = ""
        # Want dll suffix
        env["SHLIBSUFFIX"] = ".dll"

        env.Append(CCFLAGS=["-Wwrite-strings"])
        env.Append(LINKFLAGS=["-Wl,--no-undefined"])
        if env["use_static_cpp"]:
            env.Append(
                LINKFLAGS=[
                    "-static",
                    "-static-libgcc",
                    "-static-libstdc++",
                ]
            )

        # Long line hack. Use custom spawn, quick AR append (to avoid files with the same names to override each other).
        my_spawn.configure(env)

    else:
        env["use_mingw"] = True
        # Cross-compilation using MinGW
        prefix = ""
        if env["mingw_prefix"]:
            prefix = env["mingw_prefix"] + "/bin/"

        if env["arch"] == "x86_64":
            prefix += "x86_64"
        elif env["arch"] == "arm64":
            prefix += "aarch64"
        elif env["arch"] == "arm32":
            prefix += "armv7"
        elif env["arch"] == "x86_32":
            prefix += "i686"

        if env["use_llvm"]:
            env["CXX"] = prefix + "-w64-mingw32-clang++"
            env["CC"] = prefix + "-w64-mingw32-clang"
            env["AR"] = prefix + "-w64-mingw32-llvm-ar"
            env["RANLIB"] = prefix + "-w64-mingw32-ranlib"
            env["LINK"] = prefix + "-w64-mingw32-clang"
        else:
            env["CXX"] = prefix + "-w64-mingw32-g++"
            env["CC"] = prefix + "-w64-mingw32-gcc"
            env["AR"] = prefix + "-w64-mingw32-gcc-ar"
            env["RANLIB"] = prefix + "-w64-mingw32-ranlib"
            env["LINK"] = prefix + "-w64-mingw32-g++"

        # Want dll suffix
        env["SHLIBSUFFIX"] = ".dll"

        env.Append(CCFLAGS=["-Wwrite-strings"])
        env.Append(LINKFLAGS=["-Wl,--no-undefined"])
        if env["use_static_cpp"]:
            env.Append(
                LINKFLAGS=[
                    "-static",
                    "-static-libgcc",
                    "-static-libstdc++",
                ]
            )
        if env["use_llvm"]:
            env.Append(LINKFLAGS=["-lstdc++"])

        if sys.platform == "win32" or sys.platform == "msys":
            my_spawn.configure(env)

    env.Append(CPPDEFINES=["WINDOWS_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/windows/detect.py
    if env["lto"] == "auto":
        if env.get("is_msvc", False):
            # No LTO by default for MSVC, doesn't help.
            env["lto"] = "none"
        else:  # Release
            env["lto"] = "full"

    common_compiler_flags.generate(env)
