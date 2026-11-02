import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "native_loader", Path(__file__).resolve().parents[3] / "drivers/webgpu/native_loader.py"
)
assert spec and spec.loader
native_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_loader)

HEADER = """
WGPU_EXPORT void wgpuRelease(WGPUDevice device) WGPU_FUNCTION_ATTRIBUTE;
WGPU_EXPORT void const * wgpuRead(WGPUBuffer buffer, size_t offset) WGPU_FUNCTION_ATTRIBUTE;
WGPU_EXPORT WGPUFuture wgpuRequest(WGPU_NULLABLE WGPUOptions const * options, WGPUCallbackInfo callbackInfo) WGPU_FUNCTION_ATTRIBUTE;
WGPU_EXPORT WGPUInstance wgpuCreate(void) WGPU_FUNCTION_ATTRIBUTE;
"""
HELPERS = "WGPUInstance godot_webgpu_create_instance(WGPUInstanceDescriptor const * descriptor);"


class NativeLoaderTests(unittest.TestCase):
    def generate(self, platform, shared_library=False):
        return native_loader.generate_loader(HEADER, HELPERS, platform, shared_library)

    def test_forwarders_preserve_pointer_void_and_callback_arguments(self):
        for platform in ("windows", "linuxbsd", "macos"):
            with self.subTest(platform=platform):
                output = self.generate(platform)
                self.assertIn("return dawn_wgpuRelease(device);", output)
                self.assertIn("return dawn_wgpuRead(buffer, offset);", output)
                self.assertIn("return dawn_wgpuRequest(options, callbackInfo);", output)
                self.assertIn("return dawn_wgpuCreate();", output)
                self.assertIn("return dawn_godot_webgpu_create_instance(descriptor);", output)

    def test_partial_loads_reset_every_resolved_pointer(self):
        for platform, close in (("windows", "FreeLibrary"), ("linuxbsd", "dlclose"), ("macos", "dlclose")):
            with self.subTest(platform=platform):
                output = self.generate(platform)
                reset = output[
                    output.index("static void reset_native_dawn()") : output.index("static bool load_native_dawn")
                ]
                for name in ("wgpuRelease", "wgpuRead", "wgpuRequest", "wgpuCreate", "godot_webgpu_create_instance"):
                    self.assertIn(f"dawn_{name} = nullptr;", reset)
                self.assertIn(
                    f"if (!dawn_wgpuCreate) {{ reset_native_dawn(); {close}(library); return false; }}", output
                )

    def test_windows_loads_beside_the_module_containing_the_loader(self):
        for shared_library in (False, True):
            with self.subTest(shared_library=shared_library):
                output = self.generate("windows", shared_library)
                self.assertIn("GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS", output)
                self.assertIn('path_join("godot_webgpu_dawn.dll")', output)
                self.assertNotIn("dlopen", output)

    def test_unix_executables_use_the_absolute_executable_path(self):
        for platform, runtime in (("linuxbsd", "libgodot_webgpu_dawn.so"), ("macos", "libgodot_webgpu_dawn.dylib")):
            with self.subTest(platform=platform):
                output = self.generate(platform)
                self.assertIn("OS::get_singleton()->get_executable_path().get_base_dir()", output)
                self.assertNotIn("dladdr", output)
                self.assertIn(f'directory.path_join("{runtime}")', output)

    def test_unix_shared_libraries_load_beside_libgodot(self):
        for platform in ("linuxbsd", "macos"):
            with self.subTest(platform=platform):
                output = self.generate(platform, shared_library=True)
                self.assertIn("String::utf8(module.dli_fname).get_base_dir()", output)
                self.assertNotIn("get_executable_path", output)

    def test_macos_requires_the_dawn_minimum_version(self):
        self.assertIn("__builtin_available(macOS 13.0, *)", self.generate("macos"))
        self.assertNotIn("__builtin_available", self.generate("linuxbsd"))


if __name__ == "__main__":
    unittest.main()
