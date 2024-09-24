import os
import platform
import subprocess

import methods

# NOTE: The multiprocessing module is not compatible with SCons due to conflict on cPickle

# CPU architecture options.
architectures = ["x86_32", "x86_64", "arm32", "arm64", "rv64", "ppc32", "ppc64", "wasm32"]
# Aliases for CPU architectures, mapping common names to standardized options.
architecture_aliases = {
    "x86": "x86_32",
    "x64": "x86_64",
    "amd64": "x86_64",
    "armv7": "arm32",
    "armv8": "arm64",
    "arm64v8": "arm64",
    "aarch64": "arm64",
    "rv": "rv64",
    "riscv": "rv64",
    "riscv64": "rv64",
    "ppcle": "ppc32",
    "ppc": "ppc32",
    "ppc64le": "ppc64",
}

# Detect the CPU architecture of the host machine.
def detect_arch() -> str:
    host_machine = platform.machine().lower()
    if host_machine in architectures:
        return host_machine
    elif host_machine in architecture_aliases.keys():
        return architecture_aliases[host_machine]
    elif "86" in host_machine:
        # Catches x86, i386, i486, i586, i686, etc.
        return "x86_32"
    else:
        # If unsupported, defaults to x86_64 with a warning.
        methods.print_warning(f'Unsupported CPU architecture: "{host_machine}". Falling back to x86_64.')
        return "x86_64"

# Generate a build version string based on the current environment.
def get_build_version(short: bool) -> str:
    import version

    name = "custom_build"
    # Allow customization of build name from environment variable.
    if os.getenv("BUILD_NAME") is not None:
        name = os.getenv("BUILD_NAME")
    v = "%d.%d" % (version.major, version.minor)
    if version.patch > 0:
        v += ".%d" % version.patch
    status = version.status
    # Append status and name if short flag is not set.
    if not short:
        if os.getenv("GODOT_VERSION_STATUS") is not None:
            status = str(os.getenv("GODOT_VERSION_STATUS"))
        v += ".%s.%s" % (status, name)
    return v

# Combine binaries for multiple architectures into a single "fat" binary using lipo.
def lipo(prefix: str, suffix: str) -> str:
    from pathlib import Path

    target_bin = ""
    lipo_command = ["lipo", "-create"]
    arch_found = 0

    # Iterate over architectures to check for binary files.
    for arch in architectures:
        bin_name = prefix + "." + arch + suffix
        if Path(bin_name).is_file():
            target_bin = bin_name
            lipo_command += [bin_name]
            arch_found += 1

    # If more than one architecture is found, create a fat binary.
    if arch_found > 1:
        target_bin = prefix + ".fat" + suffix
        lipo_command += ["-output", target_bin]
        subprocess.run(lipo_command)

    return target_bin

# Get the path to the MoltenVK SDK for Vulkan on macOS.
def get_mvk_sdk_path(osname: str) -> str:
    def int_or_zero(i):
        # Helper function to convert a string to an integer or return 0 if invalid.
        try:
            return int(i)
        except (TypeError, ValueError):
            return 0

    def ver_parse(a):
        # Helper function to split a version string into a list of integers.
        return [int_or_zero(i) for i in a.split(".")]

    dirname = os.path.expanduser("~/VulkanSDK")
    if not os.path.exists(dirname):
        return ""

    ver_min = ver_parse("1.3.231.0")
    ver_num = ver_parse("0.0.0.0")
    files = os.listdir(dirname)
    lib_name_out = dirname
    for file in files:
        if os.path.isdir(os.path.join(dirname, file)):
            ver_comp = ver_parse(file)
            if ver_comp > ver_num and ver_comp >= ver_min:
                # Try new SDK location.
                lib_name = os.path.join(os.path.join(dirname, file), "macOS/lib/MoltenVK.xcframework/" + osname + "/")
                if os.path.isfile(os.path.join(lib_name, "libMoltenVK.a")):
                    ver_num = ver_comp
                    lib_name_out = os.path.join(os.path.join(dirname, file), "macOS/lib/MoltenVK.xcframework")
                else:
                    # Try old SDK location.
                    lib_name = os.path.join(
                        os.path.join(dirname, file), "MoltenVK/MoltenVK.xcframework/" + osname + "/"
                    )
                    if os.path.isfile(os.path.join(lib_name, "libMoltenVK.a")):
                        ver_num = ver_comp
                        lib_name_out = os.path.join(os.path.join(dirname, file), "MoltenVK/MoltenVK.xcframework")

    return lib_name_out

# Detect the installed MoltenVK SDK for Vulkan and return its path.
def detect_mvk(env, osname):
    mvk_list = [
        get_mvk_sdk_path(osname),
        "/opt/homebrew/Frameworks/MoltenVK.xcframework",
        "/usr/local/homebrew/Frameworks/MoltenVK.xcframework",
        "/opt/local/Frameworks/MoltenVK.xcframework",
    ]
    # If the Vulkan SDK path is set in the environment, prioritize it.
    if env["vulkan_sdk_path"] != "":
        mvk_list.insert(0, os.path.expanduser(env["vulkan_sdk_path"]))
        mvk_list.insert(
            0,
            os.path.join(os.path.expanduser(env["vulkan_sdk_path"]), "macOS/lib/MoltenVK.xcframework"),
        )
        mvk_list.insert(
            0,
            os.path.join(os.path.expanduser(env["vulkan_sdk_path"]), "MoltenVK/MoltenVK.xcframework"),
        )

    # Check each potential path for MoltenVK and return the first valid one.
    for mvk_path in mvk_list:
        if mvk_path and os.path.isfile(os.path.join(mvk_path, f"{osname}/libMoltenVK.a")):
            print(f"MoltenVK found at: {mvk_path}")
            return mvk_path

    return ""
