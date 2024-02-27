using Godot;

public partial class ExportDiagnosticsNodes : Node
{
    [Export]
    public Node NodeField;

    [Export]
    public Node NodeProperty { get; set; }
}

public partial class ExportDiagnosticsEmpty : Resource
{
    [Export]
    public Node {|GD0107:NodeField|};

    [Export]
    public Node {|GD0107:NodeProperty|} { get; set; }
}
