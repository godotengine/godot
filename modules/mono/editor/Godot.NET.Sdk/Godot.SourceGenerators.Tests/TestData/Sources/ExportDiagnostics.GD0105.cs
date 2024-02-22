using System;
using Godot;

public partial class ExportDiagnosticsEmpty : Node
{
    [Export]
    public int {|GD0105:this|}[int index]
    {
        get { return index; }
        set { }
    }
}
