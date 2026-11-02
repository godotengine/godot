using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace Godot.SourceGenerators.Tests;

// 2dog: hosts in other assemblies take the plugins initializer from the generated, replaceable TwoDogWebBoot.
public class GodotPluginsInitializerGeneratorTests
{
    private static async Task<(ImmutableArray<string> Generated, Compilation Output)> RunAsync(bool allowUnsafe,
        string source = "")
    {
        var references = await Constants.Net100.ResolveAsync(LanguageNames.CSharp, CancellationToken.None);
        var compilation = CSharpCompilation.Create("Game",
            new[] { CSharpSyntaxTree.ParseText(source) },
            references.Add(MetadataReference.CreateFromFile(Constants.GodotSharpAssembly.Location)),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: allowUnsafe));

        GeneratorDriver driver = CSharpGeneratorDriver.Create(new GodotPluginsInitializerGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out _);
        var generated = driver.GetRunResult().GeneratedTrees
            .Select(tree => System.IO.Path.GetFileName(tree.FilePath))
            .ToImmutableArray();
        return (generated, output);
    }

    private static void AssertNoErrors(Compilation output)
    {
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
    }

    [Fact]
    public async Task GeneratesTwoDogWebBoot()
    {
        var (generated, output) = await RunAsync(allowUnsafe: true);

        Assert.Contains("TwoDogWebBoot.generated.cs", generated);
        var boot = output.Assembly.GetTypeByMetadataName("TwoDogWebBoot");
        Assert.NotNull(boot);
        Assert.Equal(Accessibility.Public, boot!.DeclaredAccessibility);
        Assert.Single(boot.GetMembers("PluginsInitializer"));
        AssertNoErrors(output);
    }

    [Fact]
    public async Task ProjectTwoDogWebBootReplacesGeneratedOne()
    {
        const string source = """
            public static class TwoDogWebBoot
            {
                public static System.IntPtr PluginsInitializer() => System.IntPtr.Zero;
            }
            """;
        var (generated, output) = await RunAsync(allowUnsafe: true, source);

        Assert.DoesNotContain("TwoDogWebBoot.generated.cs", generated);
        Assert.Contains("GodotPlugins.Game.generated.cs", generated);
        AssertNoErrors(output);
    }

    [Fact]
    public async Task HandwrittenBootstrapStillCompiles()
    {
        // The bootstrap 2dog templates shipped before the generated accessor existed.
        const string source = """
            using System;
            using Godot.NativeInterop;

            namespace GodotPlugins.Game
            {
                internal static partial class Main
                {
                    internal static unsafe IntPtr TwoDogGetInitializePointer() =>
                        (IntPtr)(delegate* unmanaged<IntPtr, IntPtr, IntPtr, int, godot_bool>)&InitializeFromGameProject;
                }
            }

            public static class TwoDogWebBoot
            {
                public static IntPtr PluginsInitializer() => GodotPlugins.Game.Main.TwoDogGetInitializePointer();
            }
            """;
        var (generated, output) = await RunAsync(allowUnsafe: true, source);

        Assert.DoesNotContain("TwoDogWebBoot.generated.cs", generated);
        AssertNoErrors(output);
    }

    [Fact]
    public async Task SkipsTwoDogWebBootWithoutUnsafeCode()
    {
        var (generated, output) = await RunAsync(allowUnsafe: false);

        Assert.DoesNotContain("TwoDogWebBoot.generated.cs", generated);
        Assert.Contains("GodotPlugins.Game.generated.cs", generated);
        AssertNoErrors(output);
    }
}
