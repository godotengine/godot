import os
import sys
from typing import TYPE_CHECKING

from methods import print_error
from platform_methods import validate_arch

if TYPE_CHECKING:
    from SCons.Script.SConscript import SConsEnvironment


def get_name():
    return "OpenHarmony"


def can_build():
    return True


def get_tools(env: "SConsEnvironment"):
    return ["clang", "clang++", "as", "ar", "link"]


def get_opts():
    from SCons.Variables import BoolVariable

    return [
        ("OPENHARMONY_SDK_PATH", "Path to the OpenHarmony SDK", get_default_sdk_path()),
        BoolVariable(
            "generate_bundle",
            "Generate an APP bundle after building OpenHarmony binaries",
            True,
        ),
    ]


def get_doc_classes():
    return [
        "EditorExportPlatformOpenHarmony",
    ]


def get_doc_path():
    return "doc_classes"


def get_flags():
    return {
        "arch": "arm64",
        "target": "template_debug",
        "builtin_pcre2_with_jit": False,
        "opengl3": False,
    }


def get_default_sdk_path():
    return ""


def get_sdk_path(env: "SConsEnvironment"):
    sdk_root = env["OPENHARMONY_SDK_PATH"]
    if sdk_root == "":
        sdk_root = get_default_sdk_path()
    return sdk_root


def configure(env: "SConsEnvironment"):
    sdk_root = get_sdk_path(env)
    if (sdk_root == "") or (not os.path.exists(sdk_root)):
        print_error("OpenHarmony SDK not found. Please set OPENHARMONY_SDK_PATH to the SDK path.")
        sys.exit(255)

    # Validate arch.
    supported_arches = ["arm64", "x86_64"]
    validate_arch(env["arch"], get_name(), supported_arches)

    ## Compiler configuration

    # Save this in environment for use by other modules
    if sys.platform.startswith("win"):
        env["ENV"]["PATH"] = f"{sdk_root}/native/llvm/bin;" + env["ENV"]["PATH"]
    else:
        env["ENV"]["PATH"] = f"{sdk_root}/native/llvm/bin:" + env["ENV"]["PATH"]

    env["CC"] = "clang"
    env["CXX"] = "clang++"
    env["S_compiler"] = "clang"
    env["AR"] = "llvm-ar"
    env["AS"] = "llvm-as"
    env["LINK"] = "ld.lld"
    env["RANLIB"] = "llvm-ranlib"

    env.Append(
        CPPPATH=[
            f"{sdk_root}/native/llvm/lib/clang/15.0.4/include",
            f"{sdk_root}/native/llvm/include/libcxx-ohos/include/c++/v1",
        ]
    )
    target_name = ""
    if env["arch"] == "x86_64":
        target_name = "x86_64-linux-ohos"
        env.Append(ASFLAGS=["-arch", "x86_64"])
    elif env["arch"] == "arm64":
        target_name = "aarch64-linux-ohos"
        env.Append(ASFLAGS=["-arch", "aarch64"])

    env.Append(
        CCFLAGS=[
            "-fobjc-arc",
            f"--target={target_name}",
            "-fPIC",
            "-fobjc-abi-version=2",
            "-fobjc-legacy-dispatch",
            "-fmessage-length=0",
            "-fpascal-strings",
            "-fblocks",
            "-fasm-blocks",
            f"-isysroot='{sdk_root}/native/sysroot'",
        ]
    )
    env.Append(
        CXXFLAGS=[
            f"--target={target_name}",
            f"--sysroot={sdk_root}/native/sysroot/",
        ]
    )
    env.Append(
        LINKFLAGS=[
            f"--target={target_name}",
            f"--sysroot={sdk_root}/native/sysroot/",
        ]
    )
    env.Append(
        CPPPATH=[
            f"{sdk_root}/native/sysroot/usr/include/{target_name}",
            f"{sdk_root}/native/sysroot/usr/include",
            "#platform/openharmony",
        ]
    )
    env.Append(
        CPPDEFINES=[
            "OPENHARMONY_ENABLED",
            "UNIX_ENABLED",
            "__OPEN_HARMONY__",
            "MBEDTLS_NO_UDBL_DIVISION",
        ]
    )

    if env["vulkan"]:
        env.Append(CPPDEFINES=["VULKAN_ENABLED", "RD_ENABLED"])
        if not env["use_volk"]:
            env.Append(LIBS=["vulkan"])

    if env["opengl3"]:
        print_error("opengl3 is not support on OpenHarmony")
        sys.exit(255)

    env["ARGMAX"] = 8000
    env["LINKCOM"] = "$LINK @${TARGET}.rsp"
    env["SHLINKFLAGS"] = f'"--sysroot={sdk_root}/native/sysroot/" -shared -soname libgodot.so '

    def create_rsp(target, source, env):
        rsp_file = str(target[0]) + ".rsp"
        with open(rsp_file, "w") as f:
            f.write("\n".join(str(s) for s in source))
        return 0

    env["SHLIBSUFFIX"] = ".so"
    env["LINKCOM"] = env.Action(create_rsp, "Generating RSP: ${TARGET}.rsp") + env["LINKCOM"]

    env["ARCOM"] = "$AR $ARFLAGS $TARGET @${TARGET}.rsp"

    def create_ar_rsp(target, source, env):
        obj_files = [str(s) for s in source if s.get_suffix() in [".o", ".obj"]]
        rsp_file = str(target[0]) + ".rsp"
        with open(rsp_file, "w") as f:
            f.write("\n".join(obj_files))
        return 0

    env["ARCOM"] = env.Action(create_ar_rsp, "Generating AR RSP: ${TARGET}.rsp") + env["ARCOM"]
