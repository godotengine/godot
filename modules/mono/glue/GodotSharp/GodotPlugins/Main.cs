using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Godot.NativeInterop;

namespace GodotPlugins
{
    public static class Main
    {
        private static readonly List<AssemblyName> SharedAssemblies = new();
        private static readonly Assembly CoreApiAssembly = typeof(Godot.Object).Assembly;
        private static Assembly? _editorApiAssembly;

        private static readonly AssemblyLoadContext MainLoadContext =
            AssemblyLoadContext.GetLoadContext(Assembly.GetExecutingAssembly()) ??
            AssemblyLoadContext.Default;

        // Right now we do it this way for simplicity as hot-reload is disabled. It will need to be changed later.
        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Initialize(godot_bool editorHint,
            PluginsCallbacks* pluginsCallbacks, Godot.Bridge.ManagedCallbacks* managedCallbacks)
        {
            try
            {
                SharedAssemblies.Add(CoreApiAssembly.GetName());

                if (editorHint.ToBool())
                {
                    _editorApiAssembly = Assembly.Load("GodotSharpEditor");
                    SharedAssemblies.Add(_editorApiAssembly.GetName());
                }

                NativeLibrary.SetDllImportResolver(CoreApiAssembly, OnResolveDllImport);

                *pluginsCallbacks = new()
                {
                    LoadProjectAssemblyCallback = &LoadProjectAssembly,
                    LoadToolsAssemblyCallback = &LoadToolsAssembly,
                };

                *managedCallbacks = Godot.Bridge.ManagedCallbacks.Create();

                return godot_bool.True;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                *pluginsCallbacks = default;
                *managedCallbacks = default;
                return false.ToGodotBool();
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct PluginsCallbacks
        {
            public unsafe delegate* unmanaged<char*, godot_bool> LoadProjectAssemblyCallback;
            public unsafe delegate* unmanaged<char*, IntPtr> LoadToolsAssemblyCallback;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool LoadProjectAssembly(char* nAssemblyPath)
        {
            try
            {
                string assemblyPath = new(nAssemblyPath);

                var assembly = LoadPlugin(assemblyPath);

                var method = CoreApiAssembly.GetType("Godot.Bridge.ScriptManagerBridge")?
                    .GetMethod("LookupScriptsInAssembly",
                        BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);

                if (method == null)
                {
                    throw new MissingMethodException("Godot.Bridge.ScriptManagerBridge",
                        "LookupScriptsInAssembly");
                }

                method.Invoke(null, new object[] { assembly });

                return godot_bool.True;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe IntPtr LoadToolsAssembly(char* nAssemblyPath)
        {
            try
            {
                string assemblyPath = new(nAssemblyPath);

                if (_editorApiAssembly == null)
                    throw new InvalidOperationException("The Godot editor API assembly is not loaded");

                var assembly = LoadPlugin(assemblyPath);

                NativeLibrary.SetDllImportResolver(assembly, OnResolveDllImport);

                var method = assembly.GetType("GodotTools.GodotSharpEditor")?
                    .GetMethod("InternalCreateInstance",
                        BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);

                if (method == null)
                {
                    throw new MissingMethodException("GodotTools.GodotSharpEditor",
                        "InternalCreateInstance");
                }

                return (IntPtr?)method.Invoke(null, null) ?? IntPtr.Zero;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                return IntPtr.Zero;
            }
        }

        private static Assembly LoadPlugin(string assemblyPath)
        {
            string assemblyName = Path.GetFileNameWithoutExtension(assemblyPath);

            var sharedAssemblies = new List<string>();

            foreach (var sharedAssembly in SharedAssemblies)
            {
                string? sharedAssemblyName = sharedAssembly.Name;
                if (sharedAssemblyName != null)
                    sharedAssemblies.Add(sharedAssemblyName);
            }

            var loadContext = new PluginLoadContext(assemblyPath, sharedAssemblies, MainLoadContext);
            return loadContext.LoadFromAssemblyName(new AssemblyName(assemblyName));
        }

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
