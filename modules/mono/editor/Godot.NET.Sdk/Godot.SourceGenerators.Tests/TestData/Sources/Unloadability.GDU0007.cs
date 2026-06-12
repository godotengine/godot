using System.Threading;
using Godot;

// Positive: [Tool] class creating a Thread triggers GDU0007
[Tool]
public class ToolClassWithThread
{
    public void CreateThread()
    {
        var thread = {|GDU0007:new Thread(() => { })|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithThread
{
    public void CreateThread()
    {
        var thread = new Thread(() => { });
    }
}
