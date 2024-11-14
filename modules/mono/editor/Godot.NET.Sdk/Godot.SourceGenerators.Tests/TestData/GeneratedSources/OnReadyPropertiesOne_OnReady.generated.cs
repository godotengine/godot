using Godot;

#nullable enable

partial class OnReadyPropertiesOne
{
    private global::Godot.Node? _myNode;
    private global::Godot.Node MyNodeReady => _myNode ??= MyNode();
    private partial global::Godot.Node MyNode()
    {
        return GetNode<global::Godot.Node>("/Gamemanager");
    }

}
#nullable restore
