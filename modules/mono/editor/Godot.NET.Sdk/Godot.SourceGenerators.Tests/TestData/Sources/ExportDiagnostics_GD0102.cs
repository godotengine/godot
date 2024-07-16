using Godot;

public partial class ExportDiagnostics_GD0102 : Node
{
    public struct MyStruct { }

    [Export]
    public MyStruct {|GD0102:StructField|};

    [Export]
    public MyStruct {|GD0102:StructProperty|} { get; set; }
}
