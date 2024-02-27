using Godot;

public interface MyInterface
{
    public int MyProperty { get; set; }
}

public partial class ExportDiagnosticsInterface : Node, MyInterface
{
    [Export]
    public int MyProperty { get; set; }
}

public partial class ExportDiagnosticsEmpty : Node, MyInterface
{
    [Export]
    int MyInterface.{|GD0106:MyProperty|} { get; set; }
}
