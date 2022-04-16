using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Godot.Bridge;
using Godot.NativeInterop;

namespace GodotPlugins
{
    public static class Main
    {
        private static readonly List<AssemblyName> SharedAssemblies = new();
        private static readonly Assembly CoreApiAssembly = typeof(Godot.Object).Assembly;
        private static Assembly? _editorApiAssembly;
        private static Assembly? _projectAssembly;

        private static readonly AssemblyLoadContext MainLoadContext =
            AssemblyLoadContext.GetLoadContext(Assembly.GetExecutingAssembly()) ??
            AssemblyLoadContext.Default;

        private static DllImportResolver? _dllImportResolver;

        // Right now we do it this way for simplicity as hot-reload is disabled. It will need to be changed later.
        [UnmanagedCallersOnly]
        // ReSharper disable once UnusedMember.Local
        private static unsafe godot_bool InitializeFromEngine(IntPtr godotDllHandle, godot_bool editorHint,
            PluginsCallbacks* pluginsCallbacks, ManagedCallbacks* managedCallbacks, UnmanagedCallbacks* unmanagedCallbacks)
        {
            try
            {
                _dllImportResolver = new GodotDllImportResolver(godotDllHandle).OnResolveDllImport;

                SharedAssemblies.Add(CoreApiAssembly.GetName());
                NativeLibrary.SetDllImportResolver(CoreApiAssembly, _dllImportResolver);

                if (editorHint.ToBool())
                {
                    _editorApiAssembly = Assembly.Load("GodotSharpEditor");
                    SharedAssemblies.Add(_editorApiAssembly.GetName());
                    NativeLibrary.SetDllImportResolver(_editorApiAssembly, _dllImportResolver);
                }

                *pluginsCallbacks = new()
                {
                    LoadProjectAssemblyCallback = &LoadProjectAssembly,
                    LoadToolsAssemblyCallback = &LoadToolsAssembly,
                };

                *managedCallbacks = ManagedCallbacks.Create();
                NativeFuncs._unmanagedCallbacks = *unmanagedCallbacks;

                return godot_bool.True;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                return false.ToGodotBool();
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PluginsCallbacks
        {
            public unsafe delegate* unmanaged<char*, godot_bool> LoadProjectAssemblyCallback;
            public unsafe delegate* unmanaged<char*, void*, IntPtr> LoadToolsAssemblyCallback;
        }

        [UnmanagedCallersOnly]
        private static unsafe godot_bool LoadProjectAssembly(char* nAssemblyPath)
        {
            try
            {
                if (_projectAssembly != null)
                    return godot_bool.True; // Already loaded

                string assemblyPath = new(nAssemblyPath);

                _projectAssembly = LoadPlugin(assemblyPath);

                ScriptManagerBridge.LookupScriptsInAssembly(_projectAssembly);

                return godot_bool.True;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        private static unsafe IntPtr LoadToolsAssembly(char* nAssemblyPath, void* callbacks)
        {
            try
            {
                string assemblyPath = new(nAssemblyPath);

                if (_editorApiAssembly == null)
                    throw new InvalidOperationException("The Godot editor API assembly is not loaded");

                var assembly = LoadPlugin(assemblyPath);

                NativeLibrary.SetDllImportResolver(assembly, _dllImportResolver!);

                var callbacksField = assembly.GetType("GodotTools.Internals.Internal")?
                    .GetField("_unmanagedCallbacks",
                        BindingFlags.Static | BindingFlags.NonPublic);

                if (callbacksField != null)
                {
                    // TODO: Set the field value, we don't have a reference to InternalUnmanagedCallbacks
                    // defined in the GodotTools assembly which is loaded dynamically
                    // callbacksField.SetValue(null, *callbacks);
                }

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
    }
}
