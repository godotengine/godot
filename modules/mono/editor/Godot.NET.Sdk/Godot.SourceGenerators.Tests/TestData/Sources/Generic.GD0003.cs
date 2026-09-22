using Godot;

public partial class Generic<T> : GodotObject
{
    private int _field;
}

// Generic again but different generic parameters
public partial class {|GD0003:Generic|}<T, R> : GodotObject
{
    private int _field;
}

// Generic again but without generic parameters
public partial class {|GD0003:Generic|} : GodotObject
{
    private int _field;
}
