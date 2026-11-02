using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class GodotPluginsInitializerGenerator : ISourceGenerator
    {
        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotToolsProject() || context.IsGodotSourceGeneratorDisabled("GodotPluginsInitializer"))
                return;

            string source =
                @"using System;
using System.Runtime.InteropServices;
using Godot.Bridge;
using Godot.NativeInterop;

namespace GodotPlugins.Game
{
    internal static partial class Main
    {
        private static bool _resolverRegistered;
        private static bool _scriptsRegistered;

        [UnmanagedCallersOnly(EntryPoint = ""godotsharp_game_main_init"")]
        internal static godot_bool InitializeFromGameProject(IntPtr godotDllHandle, IntPtr outManagedCallbacks,
            IntPtr unmanagedCallbacks, int unmanagedCallbacksSize)
        {
            try
            {
                DllImportResolver dllImportResolver = new GodotDllImportResolver(godotDllHandle).OnResolveDllImport;

                var coreApiAssembly = typeof(global::Godot.GodotObject).Assembly;

                if (!_resolverRegistered)
                {
                    NativeLibrary.SetDllImportResolver(coreApiAssembly, dllImportResolver);
                    _resolverRegistered = true;
                }

                NativeFuncs.Initialize(unmanagedCallbacks, unmanagedCallbacksSize);

                ManagedCallbacks.Create(outManagedCallbacks);

                // 2dog: register once; hosts replay scripts after the initializer refreshes native and managed callbacks.
                if (!_scriptsRegistered)
                {
                    ScriptManagerBridge.LookupScriptsInAssembly(typeof(global::GodotPlugins.Game.Main).Assembly);
                    _scriptsRegistered = true;
                }

                return godot_bool.True;
            }
            catch (Exception e)
            {
                global::System.Console.Error.WriteLine(e);
                return false.ToGodotBool();
            }
        }

#if LIBGODOT_ENABLED
        internal static unsafe IntPtr GetInitializePointer()
        {
            return (IntPtr)(delegate* unmanaged<IntPtr, IntPtr, IntPtr, int, godot_bool>)&InitializeFromGameProject;
        }
#endif
    }
}
";

            context.AddSource("GodotPlugins.Game.generated",
                SourceText.From(source, Encoding.UTF8));

            // 2dog: libgodot hosts in other assemblies take the initializer from TwoDogWebBoot. A TwoDogWebBoot
            // declared in the project replaces this one; taking the pointer needs AllowUnsafeBlocks.
            if (context.Compilation.Options is CSharpCompilationOptions { AllowUnsafe: true }
                && context.Compilation.Assembly.GetTypeByMetadataName("TwoDogWebBoot") is null)
            {
                context.AddSource("TwoDogWebBoot.generated",
                    SourceText.From(TwoDogWebBootSource, Encoding.UTF8));
            }
        }

        private const string TwoDogWebBootSource =
            @"/// <summary>
/// Gives libgodot hosts the game assembly's plugins initializer, which registers its scripts with the engine.
/// Declare a <c>TwoDogWebBoot</c> class in the game project to replace this generated one.
/// </summary>
public static class TwoDogWebBoot
{
    public static unsafe global::System.IntPtr PluginsInitializer() =>
        (global::System.IntPtr)(delegate* unmanaged<global::System.IntPtr, global::System.IntPtr, global::System.IntPtr, int, global::Godot.NativeInterop.godot_bool>)
            &global::GodotPlugins.Game.Main.InitializeFromGameProject;
}
";
    }
}
