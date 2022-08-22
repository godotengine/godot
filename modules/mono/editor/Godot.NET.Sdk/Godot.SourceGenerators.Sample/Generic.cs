#pragma warning disable CS0169

namespace Godot.SourceGenerators.Sample
{
    partial class Generic<T> : Godot.Object
    {
        private int _field;
    }

    // Generic again but different generic parameters
    partial class Generic<T, R> : Godot.Object
    {
        private int _field;
    }

    // Generic again but without generic parameters
    partial class Generic : Godot.Object
    {
        private int _field;
    }
}
