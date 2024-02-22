using Godot;

public partial class ExportDiagnosticsEmpty : Node
{
    [Export]
    public readonly string {|GD0103:ReadOnlyField|};

    [Export]
    public string {|GD0103:ReadOnlyProperty|} { get; }
}
