using Godot;

public partial class ExportDiagnosticsEmpty : Node
{
    [Export]
    public string {|GD0104:WriteOnlyProperty|} { set { } }
}
