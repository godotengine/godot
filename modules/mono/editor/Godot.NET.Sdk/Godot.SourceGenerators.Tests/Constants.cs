using System.IO;
using System.Reflection;
using Microsoft.CodeAnalysis.Testing;

namespace Godot.SourceGenerators.Tests;

public static class Constants
{
    public static Assembly GodotSharpAssembly => typeof(GodotObject).Assembly;

    // 2dog: define the .NET 10 reference set until the test helper exposes Net100.
    public static ReferenceAssemblies Net100 => new ReferenceAssemblies(
        "net10.0",
        new PackageIdentity("Microsoft.NETCore.App.Ref", "10.0.0"),
        Path.Combine("ref", "net10.0")
    );

    public static string ExecutingAssemblyPath { get; }
    public static string SourceFolderPath { get; }
    public static string GeneratedSourceFolderPath { get; }

    static Constants()
    {
        ExecutingAssemblyPath = Path.GetFullPath(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location!)!);

        var testDataPath = Path.Combine(ExecutingAssemblyPath, "TestData");

        SourceFolderPath = Path.Combine(testDataPath, "Sources");
        GeneratedSourceFolderPath = Path.Combine(testDataPath, "GeneratedSources");
    }
}
