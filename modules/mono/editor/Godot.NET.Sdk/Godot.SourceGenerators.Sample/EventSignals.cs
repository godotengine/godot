namespace Godot.SourceGenerators.Sample;

public partial class EventSignals : GodotObject
{
    [Signal]
    public delegate void MySignalEventHandler(string str, int num);
}
