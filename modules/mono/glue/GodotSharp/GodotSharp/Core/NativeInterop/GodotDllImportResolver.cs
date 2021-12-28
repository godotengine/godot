using System;
using System.Reflection;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot.NativeInterop
{
    public static class GodotDllImportResolver
    {
        public static IntPtr OnResolveDllImport(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "__Internal")
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return Win32.GetModuleHandle(IntPtr.Zero);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return Linux.dlopen(IntPtr.Zero, Linux.RTLD_LAZY);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    return MacOS.dlopen(IntPtr.Zero, MacOS.RTLD_LAZY);
                }
            }

            return IntPtr.Zero;
        }

        // ReSharper disable InconsistentNaming
        private static class MacOS
        {
            private const string SystemLibrary = "/usr/lib/libSystem.dylib";

            public const int RTLD_LAZY = 1;

            [DllImport(SystemLibrary)]
            public static extern IntPtr dlopen(IntPtr path, int mode);
        }

        private static class Linux
        {
            // libdl.so was resulting in DllNotFoundException, for some reason...
            // libcoreclr.so should work with both CoreCLR and the .NET Core version of Mono.
            private const string SystemLibrary = "libcoreclr.so";

            public const int RTLD_LAZY = 1;

            [DllImport(SystemLibrary)]
            public static extern IntPtr dlopen(IntPtr path, int mode);
        }

        private static class Win32
        {
            private const string SystemLibrary = "Kernel32.dll";

            [DllImport(SystemLibrary)]
            public static extern IntPtr GetModuleHandle(IntPtr lpModuleName);
        }
        // ReSharper restore InconsistentNaming
    }
}
