using Godot;

public partial class OnReadyPropertiesOne : Node
{
    [OnReady("/Gamemanager")]
    private partial Node MyNode();
}
