using System;
using System.Reflection;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot.NativeInterop
{
    public class GodotDllImportResolver
    {
        private nint _internalHandle;

        public GodotDllImportResolver(nint internalHandle)
        {
            _internalHandle = internalHandle;
        }

        public nint OnResolveDllImport(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "__Internal")
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return Win32.GetModuleHandle(0);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return _internalHandle;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    return MacOS.dlopen(0, MacOS.RTLD_LAZY);
                }
            }

            return 0;
        }

        // ReSharper disable InconsistentNaming
        private static class MacOS
        {
            private const string SystemLibrary = "/usr/lib/libSystem.dylib";

            public const int RTLD_LAZY = 1;

            [DllImport(SystemLibrary)]
            public static extern nint dlopen(nint path, int mode);
        }

        private static class Win32
        {
            private const string SystemLibrary = "Kernel32.dll";

            [DllImport(SystemLibrary)]
            public static extern nint GetModuleHandle(nint lpModuleName);
        }
        // ReSharper restore InconsistentNaming
    }
}
