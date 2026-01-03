using Godot;

public abstract partial class AbstractGenericNode<[MustBeVariant] T> : Node
{
    [Export] // This should be included, but without type hints.
    public Godot.Collections.Array<T> MyArray { get; set; } = new();
}
