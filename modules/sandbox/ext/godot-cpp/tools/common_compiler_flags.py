import os
import subprocess


def using_emcc(env):
    return "emcc" in os.path.basename(env["CC"])


def using_clang(env):
    return "clang" in os.path.basename(env["CC"])


def is_vanilla_clang(env):
    if not using_clang(env):
        return False
    try:
        version = subprocess.check_output([env.subst(env["CXX"]), "--version"]).strip().decode("utf-8")
    except (subprocess.CalledProcessError, OSError):
        print("Couldn't parse CXX environment variable to infer compiler version.")
        return False
    return not version.startswith("Apple")


def exists(env):
    return True


def generate(env):
    assert env["lto"] in ["thin", "full", "none"], "Unrecognized lto: {}".format(env["lto"])
    if env["lto"] != "none":
        print("Using LTO: " + env["lto"])

    # Require C++17
    if env.get("is_msvc", False):
        env.Append(CXXFLAGS=["/std:c++17"])
    else:
        env.Append(CXXFLAGS=["-std=c++17"])

    # Disable exception handling. Godot doesn't use exceptions anywhere, and this
    # saves around 20% of binary size and very significant build time.
    if env["disable_exceptions"]:
        if env.get("is_msvc", False):
            env.Append(CPPDEFINES=[("_HAS_EXCEPTIONS", 0)])
        else:
            env.Append(CXXFLAGS=["-fno-exceptions"])
    elif env.get("is_msvc", False):
        env.Append(CXXFLAGS=["/EHsc"])

    if not env.get("is_msvc", False):
        if env["symbols_visibility"] == "visible":
            env.Append(CCFLAGS=["-fvisibility=default"])
            env.Append(LINKFLAGS=["-fvisibility=default"])
        elif env["symbols_visibility"] == "hidden":
            env.Append(CCFLAGS=["-fvisibility=hidden"])
            env.Append(LINKFLAGS=["-fvisibility=hidden"])

    # Set optimize and debug_symbols flags.
    # "custom" means do nothing and let users set their own optimization flags.
    if env.get("is_msvc", False):
        if env["debug_symbols"]:
            env.Append(CCFLAGS=["/Zi", "/FS"])
            env.Append(LINKFLAGS=["/DEBUG:FULL"])

        if env["optimize"] == "speed":
            env.Append(CCFLAGS=["/O2"])
            env.Append(LINKFLAGS=["/OPT:REF"])
        elif env["optimize"] == "speed_trace":
            env.Append(CCFLAGS=["/O2"])
            env.Append(LINKFLAGS=["/OPT:REF", "/OPT:NOICF"])
        elif env["optimize"] == "size":
            env.Append(CCFLAGS=["/O1"])
            env.Append(LINKFLAGS=["/OPT:REF"])
        elif env["optimize"] == "debug" or env["optimize"] == "none":
            env.Append(CCFLAGS=["/Od"])

        if env["lto"] == "thin":
            if not env["use_llvm"]:
                print("ThinLTO is only compatible with LLVM, use `use_llvm=yes` or `lto=full`.")
                env.Exit(255)

            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        elif env["lto"] == "full":
            if env["use_llvm"]:
                env.Append(CCFLAGS=["-flto"])
                env.Append(LINKFLAGS=["-flto"])
            else:
                env.AppendUnique(CCFLAGS=["/GL"])
                env.AppendUnique(ARFLAGS=["/LTCG"])
                env.AppendUnique(LINKFLAGS=["/LTCG"])
    else:
        if env["debug_symbols"]:
            # Adding dwarf-4 explicitly makes stacktraces work with clang builds,
            # otherwise addr2line doesn't understand them.
            env.Append(CCFLAGS=["-gdwarf-4"])
            if using_emcc(env):
                # Emscripten only produces dwarf symbols when using "-g3".
                env.AppendUnique(CCFLAGS=["-g3"])
                # Emscripten linker needs debug symbols options too.
                env.AppendUnique(LINKFLAGS=["-gdwarf-4"])
                env.AppendUnique(LINKFLAGS=["-g3"])
            elif env.dev_build:
                env.Append(CCFLAGS=["-g3"])
            else:
                env.Append(CCFLAGS=["-g2"])
        else:
            if using_clang(env) and not is_vanilla_clang(env) and not env["use_mingw"]:
                # Apple Clang, its linker doesn't like -s.
                env.Append(LINKFLAGS=["-Wl,-S", "-Wl,-x", "-Wl,-dead_strip"])
            else:
                env.Append(LINKFLAGS=["-s"])

        if env["optimize"] == "speed":
            env.Append(CCFLAGS=["-O3"])
        # `-O2` is friendlier to debuggers than `-O3`, leading to better crash backtraces.
        elif env["optimize"] == "speed_trace":
            env.Append(CCFLAGS=["-O2"])
        elif env["optimize"] == "size":
            env.Append(CCFLAGS=["-Os"])
        elif env["optimize"] == "debug":
            env.Append(CCFLAGS=["-Og"])
        elif env["optimize"] == "none":
            env.Append(CCFLAGS=["-O0"])

        if env["lto"] == "thin":
            if (env["platform"] == "windows" or env["platform"] == "linux") and not env["use_llvm"]:
                print("ThinLTO is only compatible with LLVM, use `use_llvm=yes` or `lto=full`.")
                env.Exit(255)
            env.Append(CCFLAGS=["-flto=thin"])
            env.Append(LINKFLAGS=["-flto=thin"])
        elif env["lto"] == "full":
            env.Append(CCFLAGS=["-flto"])
            env.Append(LINKFLAGS=["-flto"])
