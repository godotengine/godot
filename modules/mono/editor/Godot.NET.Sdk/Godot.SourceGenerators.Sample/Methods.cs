using System.Diagnostics.CodeAnalysis;

namespace Godot.SourceGenerators.Sample;

[SuppressMessage("ReSharper", "RedundantNameQualifier")]
public partial class Methods : GodotObject
{
    private void MethodWithOverload()
    {
    }

    private void MethodWithOverload(int a)
    {
    }

    private void MethodWithOverload(int a, int b)
    {
    }

    // Should be ignored. The previous one is picked.
    // ReSharper disable once UnusedMember.Local
    private void MethodWithOverload(float a, float b)
    {
    }

    // Generic methods should be ignored.
    // ReSharper disable once UnusedMember.Local
    private void GenericMethod<T>(T t)
    {
    }
}
