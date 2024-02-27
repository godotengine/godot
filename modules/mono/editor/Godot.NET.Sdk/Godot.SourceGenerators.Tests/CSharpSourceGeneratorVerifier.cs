using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Testing;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.CodeAnalysis.Testing.Verifiers;
using Microsoft.CodeAnalysis.Text;

namespace Godot.SourceGenerators.Tests;

public record VerifierConfiguration
{
    public CompilerDiagnostics CompilerDiagnostics { get; init; } = CompilerDiagnostics.Errors;

    public string[] Sources { get; init; } = Array.Empty<string>();
    public string[] GeneratedSources { get; init; } = Array.Empty<string>();

    public string[] DisabledGenerators { get; init; } = Array.Empty<string>();
}

public static class CSharpSourceGeneratorVerifier<TSourceGenerator>
where TSourceGenerator : ISourceGenerator, new()
{
    public class Test : CSharpSourceGeneratorTest<TSourceGenerator, XUnitVerifier>
    {
        public Test()
        {
            ReferenceAssemblies = ReferenceAssemblies.Net.Net60;

            SolutionTransforms.Add((Solution solution, ProjectId projectId) =>
            {
                Project project = solution.GetProject(projectId)!.AddMetadataReference(
                    Constants.GodotSharpAssembly.CreateMetadataReference()
                );

                return project.Solution;
            });
        }

        public Test WithSources(params string[] filenames)
        {
            return WithSources(filenames.Select(filename => (
                Filename: filename,
                ContentFilename: filename
            )).ToArray());
        }

        public Test WithSources(params (string Filename, string ContentFilename)[] filenames)
        {
            TestState.Sources.AddRange(filenames.Select(filename => (
                filename: filename.Filename,
                content: SourceTextFromFile(Path.Combine(Constants.SourceFolderPath, filename.ContentFilename))
            )));

            return this;
        }

        public Test WithGeneratedSources(params string[] filenames)
        {
            return WithGeneratedSources(filenames.Select(filename => (
                Filename: FullGeneratedSourceFileName(filename),
                ContentFilename: filename
            )).ToArray());
        }

        public Test WithGeneratedSources(params (string Filename, string ContentFilename)[] filenames)
        {
            TestState.GeneratedSources.AddRange(filenames.Select(filename => (
                filename: filename.Filename,
                content: SourceTextFromFile(Path.Combine(Constants.GeneratedSourceFolderPath, filename.ContentFilename))
            )));

            return this;
        }
    }

    public static Task Verify(string source, string generatedSource)
    {
        return MakeVerifier().WithSources(source).WithGeneratedSources(generatedSource).RunAsync();
    }

    public static Test MakeVerifier(VerifierConfiguration? verifierConfiguration = null)
    {
        verifierConfiguration ??= new VerifierConfiguration();

        var verifier = new Test();

        verifier.TestState.AnalyzerConfigFiles.Add((Constants.GlobalConfig, $"""
        is_global = true
        build_property.GodotProjectDir = {Constants.ExecutingAssemblyPath}
        build_property.GodotDisabledSourceGenerators = {string.Join(";", verifierConfiguration.DisabledGenerators)}
        """));

        verifier.CompilerDiagnostics = verifierConfiguration.CompilerDiagnostics;

        verifier.WithSources(verifierConfiguration.Sources);
        verifier.WithGeneratedSources(verifierConfiguration.GeneratedSources);

        return verifier;
    }

    private static SourceText SourceTextFromFile(string filePath)
    {
        return SourceText.From(File.ReadAllText(filePath), Encoding.UTF8);
    }

    private static string FullGeneratedSourceFileName(string name)
    {
        var generatorType = typeof(TSourceGenerator);
        return Path.Combine(generatorType.Namespace!, generatorType.FullName!, name);
    }
}
