using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.Loader;

namespace GodotPlugins
{
    public class PluginLoadContext : AssemblyLoadContext
    {
        private readonly AssemblyDependencyResolver _resolver;
        private readonly ICollection<string> _sharedAssemblies;
        private readonly AssemblyLoadContext _mainLoadContext;

        public string? AssemblyLoadedPath { get; private set; }

        public PluginLoadContext(string pluginPath, ICollection<string> sharedAssemblies,
            AssemblyLoadContext mainLoadContext, bool isCollectible)
            : base(isCollectible)
        {
            _resolver = new AssemblyDependencyResolver(pluginPath);
            _sharedAssemblies = sharedAssemblies;
            _mainLoadContext = mainLoadContext;

            if (string.IsNullOrEmpty(AppContext.BaseDirectory))
            {
                // See https://github.com/dotnet/runtime/blob/v6.0.0/src/libraries/System.Private.CoreLib/src/System/AppContext.AnyOS.cs#L17-L35
                // but Assembly.Location is unavailable, because we load assemblies from memory.
                string? baseDirectory = Path.GetDirectoryName(pluginPath);
                if (baseDirectory != null)
                {
                    if (!Path.EndsInDirectorySeparator(baseDirectory))
                        baseDirectory += Path.DirectorySeparatorChar;
                    // This SetData call effectively sets AppContext.BaseDirectory
                    // See https://github.com/dotnet/runtime/blob/v6.0.0/src/libraries/System.Private.CoreLib/src/System/AppContext.cs#L21-L25
                    AppDomain.CurrentDomain.SetData("APP_CONTEXT_BASE_DIRECTORY", baseDirectory);
                }
                else
                {
                    // TODO: How to log from GodotPlugins? (delegate pointer?)
                    Console.Error.WriteLine("Failed to set AppContext.BaseDirectory. Dynamic loading of libraries may fail.");
                }
            }
        }

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            if (assemblyName.Name == null)
                return null;

            if (_sharedAssemblies.Contains(assemblyName.Name))
                return _mainLoadContext.LoadFromAssemblyName(assemblyName);

            string? assemblyPath = _resolver.ResolveAssemblyToPath(assemblyName);
            if (assemblyPath != null)
            {
                AssemblyLoadedPath = assemblyPath;

                // Load in memory to prevent locking the file
                using var assemblyFile = File.Open(assemblyPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                string pdbPath = Path.ChangeExtension(assemblyPath, ".pdb");

                if (File.Exists(pdbPath))
                {
                    using var pdbFile = File.Open(pdbPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                    return LoadFromStream(assemblyFile, pdbFile);
                }

                return LoadFromStream(assemblyFile);
            }

            return null;
        }

        protected override IntPtr LoadUnmanagedDll(string unmanagedDllName)
        {
            string? libraryPath = _resolver.ResolveUnmanagedDllToPath(unmanagedDllName);
            if (libraryPath != null)
                return LoadUnmanagedDllFromPath(libraryPath);

            return IntPtr.Zero;
        }
    }
}
