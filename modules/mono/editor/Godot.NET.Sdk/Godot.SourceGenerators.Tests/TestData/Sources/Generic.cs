using Godot;
using Godot.Collections;

public partial class Generic<[MustBeVariant] T> : GodotObject
{
    [Export] public T RegularField;
    [Export] public T RegularProperty { get; set; }

    [Export] public Array<T> ArrayProperty { get; set; }

    [Signal]
    public delegate T GenericSignalEventHandler(T var);

    public T GenericMethod(T var)
    {
        return var;
    }
}
