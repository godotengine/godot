"""Configure the optional native Dawn dependency for desktop platforms."""

from pathlib import Path

from methods import print_error


def configure_native_webgpu(env):
    if not env["webgpu"]:
        return
    root = Path(env["webgpu_native_path"]).resolve()
    runtime_names = {
        "windows": "bin/godot_webgpu_dawn.dll",
        "linuxbsd": "lib/libgodot_webgpu_dawn.so",
        "macos": "lib/libgodot_webgpu_dawn.dylib",
    }
    runtime = root / runtime_names[env["platform"]]
    if not (root / "include/dawn/webgpu.h").is_file() or not runtime.is_file():
        print_error(
            "Native WebGPU requires webgpu_native_path=<installed Dawn boundary SDK>. See drivers/webgpu/native/README.md."
        )
        env.Exit(255)
    env.AppendUnique(CPPDEFINES=["WEBGPU_ENABLED", "RD_ENABLED"])
    env.Prepend(CPPPATH=[str(root / "include")])
    env["webgpu_native_path"] = str(root)  # SCsub resolves relative paths from its own directory.
    env["webgpu_native_runtimes"] = [str(runtime)]
    if env["platform"] == "windows":
        for name in ("dxcompiler.dll", "dxil.dll"):
            compiler = root / "bin" / name
            if compiler.is_file():
                env["webgpu_native_runtimes"].append(str(compiler))
