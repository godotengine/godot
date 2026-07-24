using Godot;

// This raises a GD0004 diagnostic error: classes inheriting from GodotObject must declare a parameterless constructor
public partial class {|GD0004:CustomParameterlessConstructorClass|} : GodotObject
{
    public CustomParameterlessConstructorClass(int value)
    {
        _ = value;
    }
}
