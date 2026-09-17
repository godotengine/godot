using System;
using System.Text;
using Microsoft.CodeAnalysis;
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

            bool supportLegacyNonTrimSafeApis = IsGodotSupportLegacyNonTrimSafeApisEnabled(context);

            var source = new StringBuilder();

            source.Append(
                    @"using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Godot.Bridge;
using Godot.NativeInterop;

namespace GodotPlugins.Game
{
    internal static partial class Main
    {
        public static partial void RegisterScriptTypes();

#if TOOLS
        [RequiresUnreferencedCode(""TOOLS build of Godot project is not compatible with trimming"")]
#endif
        internal static godot_bool Initialize(IntPtr godotDllHandle, IntPtr outManagedCallbacks,
            IntPtr unmanagedCallbacks, int unmanagedCallbacksSize)
        {
            DllImportResolver dllImportResolver = new GodotDllImportResolver(godotDllHandle).OnResolveDllImport;

            var coreApiAssembly = typeof(global::Godot.GodotObject).Assembly;

            NativeLibrary.SetDllImportResolver(coreApiAssembly, dllImportResolver);

            NativeFuncs.Initialize(unmanagedCallbacks, unmanagedCallbacksSize);

#if TOOLS
            ManagedCallbacks.CreateForToolsBuild(outManagedCallbacks);
#else
")
                .Append("            ")
                .Append(supportLegacyNonTrimSafeApis
                    ? "ManagedCallbacks.CreateIncludingLegacyCallbacks(outManagedCallbacks);"
                    : "ManagedCallbacks.CreateExcludingLegacyCallbacks(outManagedCallbacks);"
                )
                .Append(
                    @"
#endif

            ScriptManagerBridge.InitializeNativeClassConstructors();
");

            if (supportLegacyNonTrimSafeApis)
            {
                source.Append(
                    @"
            // Use of legacy obsolete API manually enabled with $(GodotSupportLegacyNonTrimSafeAPIs) in MSBuild.
#pragma warning disable CS0618 // Type or member is obsolete
            ScriptManagerBridge.EnableLegacyScriptTypeMetaResolver();
#pragma warning restore CS0618 // Type or member is obsolete
");
            }

            source.Append(
                @"
            RegisterScriptTypes();

            return godot_bool.True;
        }

#if TOOLS
        [RequiresUnreferencedCode(""TOOLS build of Godot project is not compatible with trimming"")]
#endif
        [UnmanagedCallersOnly(EntryPoint = ""godotsharp_game_main_init"")]
        private static godot_bool InitializeFromGameProject(IntPtr godotDllHandle, IntPtr outManagedCallbacks,
            IntPtr unmanagedCallbacks, int unmanagedCallbacksSize)
        {
            try
            {
                return Initialize(godotDllHandle, outManagedCallbacks, unmanagedCallbacks, unmanagedCallbacksSize);
            }
            catch (Exception e)
            {
                global::System.Console.Error.WriteLine(e);
                return false.ToGodotBool();
            }
        }
    }
}
");

            context.AddSource("GodotPlugins.Game.generated",
                SourceText.From(source.ToString(), Encoding.UTF8));
        }

        private static bool IsGodotSupportLegacyNonTrimSafeApisEnabled(GeneratorExecutionContext context)
            => context.TryGetGlobalAnalyzerProperty("GodotSupportLegacyNonTrimSafeAPIs", out string? toggle) &&
               toggle != null &&
               toggle.Equals("true", StringComparison.OrdinalIgnoreCase);
    }
}
