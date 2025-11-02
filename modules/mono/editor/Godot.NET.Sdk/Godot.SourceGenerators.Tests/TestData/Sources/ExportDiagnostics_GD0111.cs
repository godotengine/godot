using Godot;
using Godot.Collections;

[Tool]
public partial class ExportDiagnostics_GD0111 : Node
{
    private Callable _backingField;

    [ExportToolButton("")]
    public Callable {|GD0111:MyButtonGet|} { get; }

    [ExportToolButton("")]
    public Callable {|GD0111:MyButtonGetSet|} { get; set; }

    [ExportToolButton("")]
    public Callable {|GD0111:MyButtonGetWithBackingField|} { get => _backingField; }

    [ExportToolButton("")]
    public Callable {|GD0111:MyButtonGetSetWithBackingField|} { get => _backingField; set => _backingField = value; }

    [ExportToolButton("")]
    public Callable MyButtonOkWithCallableCreationExpression => new Callable(this, "");

    [ExportToolButton("")]
    public Callable MyButtonOkWithImplicitCallableCreationExpression => new(this, "");

    [ExportToolButton("")]
    public Callable MyButtonOkWithCallableFromExpression => Callable.From(null);
}
