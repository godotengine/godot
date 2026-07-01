using Godot;

// Positive: [Tool] class creating System.Threading.Timer triggers GDU0008
[Tool]
public class ToolClassWithThreadingTimer
{
    public void CreateTimer()
    {
        var timer = {|GDU0008:new System.Threading.Timer(_ => { }, null, 0, 1000)|};
    }
}

// Positive (conservative): [Tool] class creating System.Timers.Timer triggers GDU0008
// because it is typically followed by Elapsed += handler from the collectible assembly.
// This is a known conservative heuristic; suppress if no collectible callback is attached.
[Tool]
public class ToolClassWithTimersTimer
{
    public void CreateTimer()
    {
        var timer = {|GDU0008:new System.Timers.Timer(1000)|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithTimer
{
    public void CreateTimer()
    {
        var timer = new System.Threading.Timer(_ => { }, null, 0, 1000);
    }
}
