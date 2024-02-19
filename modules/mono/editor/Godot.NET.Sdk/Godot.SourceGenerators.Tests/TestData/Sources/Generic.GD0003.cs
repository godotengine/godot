using Godot;

partial class Generic<T> : GodotObject
{
    private int _field;
}

// Generic again but different generic parameters
partial class {|GD0003:Generic|}<T, R> : GodotObject
{
    private int _field;
}

// Generic again but without generic parameters
partial class {|GD0003:Generic|} : GodotObject
{
    private int _field;
}
