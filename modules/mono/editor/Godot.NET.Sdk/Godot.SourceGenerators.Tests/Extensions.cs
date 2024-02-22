using System.Reflection;
using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators.Tests;

internal static class Extensions
{
    internal static MetadataReference CreateMetadataReference(this Assembly assembly)
    {
        return MetadataReference.CreateFromFile(assembly.Location);
    }
}
