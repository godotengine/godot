namespace Godot.SourceGenerators.Sample
{
    public partial class ScriptBoilerplate : Godot.Node
    {
        private NodePath _nodePath;
        private int _velocity;

        public override void _Process(float delta)
        {
            _ = delta;

            base._Process(delta);
        }

        public int Bazz(StringName name)
        {
            _ = name;
            return 1;
        }
    }
}
