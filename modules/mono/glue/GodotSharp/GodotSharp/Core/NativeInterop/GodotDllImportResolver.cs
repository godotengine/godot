using System;
using System.Reflection;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot.NativeInterop
{
    /// <summary>
    /// Provides a resolver for handling dll imports from Godot.
    /// </summary>
    public class GodotDllImportResolver
    {
        private IntPtr _internalHandle;

        /// <summary>
        /// Constructs a new <see cref="GodotDllImportResolver"/> using the provided handle.
        /// </summary>
        /// <param name="internalHandle">The pointer of the handle to resolve.</param>
        public GodotDllImportResolver(IntPtr internalHandle)
        {
            _internalHandle = internalHandle;
        }

        /// <summary>
        /// Called when the dll import process is resolved.
        /// </summary>
        /// <param name="libraryName">The library name for this dll.</param>
        /// <param name="assembly">The assembly context of this dll.</param>
        /// <param name="searchPath">The path to search.</param>
        /// <returns>The internal handle of the resolved dll.</returns>
        public IntPtr OnResolveDllImport(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName == "__Internal")
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    return Win32.GetModuleHandle(IntPtr.Zero);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    return _internalHandle;
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

        private static class Win32
        {
            private const string SystemLibrary = "Kernel32.dll";

            [DllImport(SystemLibrary)]
            public static extern IntPtr GetModuleHandle(IntPtr lpModuleName);
        }
        // ReSharper restore InconsistentNaming
    }
}
