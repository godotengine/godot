using Godot;

#nullable enable

partial class OnReadyPropertiesTwo
{
    private global::Godot.Node? _myNode;
    private global::Godot.Node MyNodeReady => _myNode ??= MyNode();
    private partial global::Godot.Node MyNode()
    {
        return GetNode<global::Godot.Node>("/Gamemanager");
    }

    private global::Godot.Node? _myNode2;
    private global::Godot.Node? MyNode2Ready => _myNode2 ??= MyNode2();
    private partial global::Godot.Node? MyNode2()
    {
        return GetNodeOrNull<global::Godot.Node>("/Gamemanager2");
    }

    private global::Godot.Node? _myNode3;
    private partial global::Godot.Node MyNode3 => _myNode3 ??= GetNode<global::Godot.Node>("/Gamemanager3");

    private global::Godot.Node? _myNode4;
    private partial global::Godot.Node? MyNode4 => _myNode4 ??= GetNodeOrNull<global::Godot.Node>("/Gamemanager4");

}
#nullable restore
