import common_compiler_flags
from SCons.Tool import clang, clangxx
from SCons.Variables import BoolVariable


def options(opts):
    opts.Add(BoolVariable("use_llvm", "Use the LLVM compiler - only effective when targeting Linux", False))
    opts.Add(BoolVariable("use_static_cpp", "Link libgcc and libstdc++ statically for better portability", True))


def exists(env):
    return True


def generate(env):
    if env["use_llvm"]:
        clang.generate(env)
        clangxx.generate(env)
    elif env.use_hot_reload:
        # Required for extensions to truly unload.
        env.Append(CXXFLAGS=["-fno-gnu-unique"])

    env.Append(CCFLAGS=["-fPIC", "-Wwrite-strings"])
    env.Append(LINKFLAGS=["-Wl,-R,'$$ORIGIN'"])

    if env["arch"] == "x86_64":
        # -m64 and -m32 are x86-specific already, but it doesn't hurt to
        # be clear and also specify -march=x86-64. Similar with 32-bit.
        env.Append(CCFLAGS=["-m64", "-march=x86-64"])
        env.Append(LINKFLAGS=["-m64", "-march=x86-64"])
    elif env["arch"] == "x86_32":
        env.Append(CCFLAGS=["-m32", "-march=i686"])
        env.Append(LINKFLAGS=["-m32", "-march=i686"])
    elif env["arch"] == "arm64":
        env.Append(CCFLAGS=["-march=armv8-a"])
        env.Append(LINKFLAGS=["-march=armv8-a"])
    elif env["arch"] == "rv64":
        env.Append(CCFLAGS=["-march=rv64gc"])
        env.Append(LINKFLAGS=["-march=rv64gc"])

    # Link statically for portability
    if env["use_static_cpp"]:
        env.Append(LINKFLAGS=["-static-libgcc", "-static-libstdc++"])

    env.Append(CPPDEFINES=["LINUX_ENABLED", "UNIX_ENABLED"])

    # Refer to https://github.com/godotengine/godot/blob/master/platform/linuxbsd/detect.py
    if env["lto"] == "auto":
        env["lto"] = "full"

    common_compiler_flags.generate(env)
