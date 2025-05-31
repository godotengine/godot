using Godot;

public partial class EventSignals : GodotObject
{
    [Signal]
    public delegate void MySignalEventHandler(string str, int num);

    private struct MyStruct { }

    [Signal]
    private delegate void {|GD0201:MyInvalidSignal|}();

    [Signal]
    private delegate void MyInvalidParameterTypeSignalEventHandler(MyStruct {|GD0202:myStruct|});

    [Signal]
    private delegate MyStruct {|GD0203:MyInvalidReturnTypeSignalEventHandler|}();
}
