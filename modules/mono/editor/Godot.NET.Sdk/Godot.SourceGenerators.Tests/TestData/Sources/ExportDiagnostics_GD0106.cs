using Godot;

public interface MyInterface
{
    public int MyProperty { get; set; }
}

public partial class ExportDiagnostics_GD0106_OK : Node, MyInterface
{
    [Export]
    public int MyProperty { get; set; }
}

public partial class ExportDiagnostics_GD0106_KO : Node, MyInterface
{
    [Export]
    int MyInterface.{|GD0106:MyProperty|} { get; set; }
}
