#pragma warning disable CS0169

namespace Godot.SourceGenerators.Sample
{
    public partial class ScriptBoilerplate : Node
    {
        private NodePath _nodePath;
        private int _velocity;

        public override void _Process(double delta)
        {
            _ = delta;

            base._Process(delta);
        }

        public int Bazz(StringName name)
        {
            _ = name;
            return 1;
        }

        public void IgnoreThisMethodWithByRefParams(ref int a)
        {
            _ = a;
        }
    }

    public partial struct OuterClass
    {
        public partial class NesterClass : RefCounted
        {
            public override Variant _Get(StringName property) => default;
        }
    }
}
