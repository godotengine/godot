namespace Godot.SourceGenerators.Sample
{
    partial class Generic<T> : Godot.Object
    {
    }

    // Generic again but different generic parameters
    partial class Generic<T, R> : Godot.Object
    {
    }

    // Generic again but without generic parameters
    partial class Generic : Godot.Object
    {
    }
}
