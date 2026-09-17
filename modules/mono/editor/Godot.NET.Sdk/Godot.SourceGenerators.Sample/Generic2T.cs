#pragma warning disable CS0169

namespace Godot.SourceGenerators.Sample
{
    // Generic again but different generic parameters
    public partial class Generic2T<T, R> : GodotObject
    {
        private int _field;

        public partial class NestedGenericClassWithoutTypeParameters : GodotObject;
        private partial class PrivateNestedGenericClassWithoutTypeParameters : GodotObject;
    }
}
