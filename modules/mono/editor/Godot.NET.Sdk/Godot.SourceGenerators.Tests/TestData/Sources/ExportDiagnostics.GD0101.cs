using Godot;

public partial class ExportDiagnosticsEmpty : Node
{
    [Export]
    public static string {|GD0101:StaticField|};

    [Export]
    public static int {|GD0101:StaticProperty|} { get; set; }
}
