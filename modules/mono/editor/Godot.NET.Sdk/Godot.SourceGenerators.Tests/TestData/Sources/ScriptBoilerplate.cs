using Godot;

public partial class ScriptBoilerplate : Node
{
    private NodePath _nodePath;
    private int _velocity;

    [IgnoreMember]
    private int _ignoredField;

    [IgnoreMember]
    private int IgnoredProperty { get; set; }

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
    public partial class NestedClass : RefCounted
    {
        public override Variant _Get(StringName property) => default;
    }
}
