using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators
{
    [Generator]
    public class WebHostGenerator : ISourceGenerator
    {
        private static readonly DiagnosticDescriptor MissingPluginsInitializerRule =
            new DiagnosticDescriptor(id: "GDWEB0003",
                title: "The Godot web .NET host requires the GodotPluginsInitializer generator",
                messageFormat: "The Godot web .NET host requires the GodotPluginsInitializer generator, which is disabled",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The Godot web .NET host requires the GodotPluginsInitializer generator. Remove 'GodotPluginsInitializer' from the 'GodotDisabledSourceGenerators' build property.");

        private static readonly DiagnosticDescriptor MissingRootNamespaceRule =
            new DiagnosticDescriptor(id: "GDWEB0001",
                title: "Missing RootNamespace for Godot web .NET host",
                messageFormat: "The Godot web .NET host generator requires the 'RootNamespace' build property to be set",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The Godot web .NET host generator requires the 'RootNamespace' build property to be set.");

        private static readonly DiagnosticDescriptor InvalidRootNamespaceRule =
            new DiagnosticDescriptor(id: "GDWEB0002",
                title: "Invalid RootNamespace for Godot web .NET host",
                messageFormat: "The 'RootNamespace' value '{0}' is not a valid C# namespace",
                category: "Usage",
                DiagnosticSeverity.Error,
                isEnabledByDefault: true,
                "The 'RootNamespace' value is not a valid C# namespace. Each dot-separated segment must be a valid C# identifier.");

        public void Initialize(GeneratorInitializationContext context)
        {
        }

        public void Execute(GeneratorExecutionContext context)
        {
            if (context.IsGodotToolsProject() || context.IsGodotSourceGeneratorDisabled("WebHost"))
                return;

            if (!context.TryGetGlobalAnalyzerProperty("GodotTargetPlatform", out string? targetPlatform) ||
                targetPlatform != "web")
                return;

            if (context.IsGodotSourceGeneratorDisabled("GodotPluginsInitializer"))
            {
                context.ReportDiagnostic(Diagnostic.Create(MissingPluginsInitializerRule, Location.None));
                return;
            }

            // ReSharper disable once ReplaceWithStringIsNullOrEmpty
            if (!context.TryGetGlobalAnalyzerProperty("RootNamespace", out string? rootNamespace) ||
                rootNamespace == null || rootNamespace.Length == 0)
            {
                context.ReportDiagnostic(Diagnostic.Create(MissingRootNamespaceRule, Location.None));
                return;
            }

            if (!TryEscapeNamespace(rootNamespace, out string escapedNamespace))
            {
                context.ReportDiagnostic(Diagnostic.Create(InvalidRootNamespaceRule, Location.None, rootNamespace));
                return;
            }

            context.AddSource("LibGodot.Web.generated",
                SourceText.From(EmitInteropSource(escapedNamespace), Encoding.UTF8));
        }

        private static bool TryEscapeNamespace(string candidate, out string escaped)
        {
            string[] segments = candidate.Split('.');
            string[] escapedSegments = new string[segments.Length];

            for (int i = 0; i < segments.Length; i++)
            {
                string segment = segments[i];

                if (!SyntaxFacts.IsValidIdentifier(segment))
                {
                    escaped = string.Empty;
                    return false;
                }

                escapedSegments[i] = SyntaxFacts.GetKeywordKind(segment) != SyntaxKind.None ?
                    "@" + segment :
                    segment;
            }

            escaped = string.Join(".", escapedSegments);
            return true;
        }

        private static string EmitInteropSource(string hostNamespace)
        {
            return @"#nullable enable

using System;
using System.Runtime.InteropServices;

namespace " + hostNamespace + @"
{
    internal static unsafe class LibGodot
    {
        [DllImport(""*"", EntryPoint = ""libgodot_create_godot_instance"")]
        internal static extern IntPtr CreateGodotInstance(int argc, byte** argv, IntPtr initializationFunction);

        [DllImport(""*"", EntryPoint = ""libgodot_web_start_godot_instance"")]
        [return: MarshalAs(UnmanagedType.U1)]
        internal static extern bool StartGodotInstance(IntPtr godotInstance);

        [DllImport(""*"", EntryPoint = ""libgodot_destroy_godot_instance"")]
        internal static extern void DestroyGodotInstance(IntPtr godotInstance);

        [DllImport(""*"", EntryPoint = ""libgodot_web_register_godot_plugins_initialize"")]
        internal static extern void RegisterGodotPluginsInitialize(IntPtr godotPluginsInitialize);
    }

    internal static unsafe class GodotHost
    {
        internal static IntPtr Run(string[] args, IntPtr initializationFunction)
        {
            int argc = args.Length + 1;
            byte** argv = (byte**)NativeMemory.Alloc((nuint)((argc + 1) * sizeof(byte*)));
            argv[0] = (byte*)Marshal.StringToCoTaskMemUTF8(AppDomain.CurrentDomain.FriendlyName);
            for (int i = 0; i < args.Length; i++)
            {
                argv[i + 1] = (byte*)Marshal.StringToCoTaskMemUTF8(args[i]);
            }
            argv[argc] = null;

            IntPtr godotInstance = LibGodot.CreateGodotInstance(argc, argv, initializationFunction);

            // Godot copies argv into its own command-line storage during creation, so the
            // native strings and the pointer array can be released as soon as it returns.
            for (int i = 0; i < argc; i++)
            {
                Marshal.FreeCoTaskMem((IntPtr)argv[i]);
            }
            NativeMemory.Free(argv);

            if (godotInstance != IntPtr.Zero && !LibGodot.StartGodotInstance(godotInstance))
            {
                LibGodot.DestroyGodotInstance(godotInstance);
                return IntPtr.Zero;
            }

            return godotInstance;
        }
    }

    internal static class GodotWebEntryPoint
    {
        public static int Main(string[] args)
        {
            LibGodot.RegisterGodotPluginsInitialize(global::GodotPlugins.Game.Main.GetGodotPluginsInitializeFunctionPointer());
            IntPtr godotInstance = GodotHost.Run(args, IntPtr.Zero);
            return godotInstance == IntPtr.Zero ? 1 : 0;
        }
    }
}

namespace GodotPlugins.Game
{
    internal static partial class Main
    {
        internal static unsafe IntPtr GetGodotPluginsInitializeFunctionPointer()
        {
            return (IntPtr)(delegate* unmanaged<IntPtr, IntPtr, IntPtr, int, global::Godot.NativeInterop.godot_bool>)&InitializeFromGameProject;
        }
    }
}
";
        }
    }
}
