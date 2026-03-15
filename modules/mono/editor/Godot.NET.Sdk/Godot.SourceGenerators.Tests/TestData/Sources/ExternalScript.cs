using Godot;

namespace ExternalModule;

public partial class ExternalScript : Node
{
    private int _health;

    public override void _Ready()
    {
        _health = 100;
    }
}
