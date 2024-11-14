using Godot;

public partial class OnReadyPropertiesTwo : Node
{
    [OnReady("/Gamemanager")]
    private partial Node MyNode();

    [OnReady("/Gamemanager2")]
    private partial Node? MyNode2();

    [OnReady("/Gamemanager3")]
    private partial Node MyNode3 { get; }

    [OnReady("/Gamemanager4")]
    private partial Node? MyNode4 { get; }
}
