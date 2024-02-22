using Godot;

public partial class ExportDiagnosticsEmpty : Node
{
    public struct MyStruct { }

    [Export]
    public MyStruct {|GD0102:StructField|};

    [Export]
    public MyStruct {|GD0102:StructProperty|} { get; set; }
}
