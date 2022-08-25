namespace Godot.SourceGenerators.Sample;

public partial class EventSignals : Godot.Object
{
    [Signal]
    public delegate void MySignalEventHandler(string str, int num);
}
