using Godot;

partial class Generic<T> : GodotObject
{
    private int _field;
}

// Generic again but different generic parameters
partial class Generic<T, R> : GodotObject
{
    private int _field;
}

// Generic again but without generic parameters
partial class Generic : GodotObject
{
    private int _field;
}
