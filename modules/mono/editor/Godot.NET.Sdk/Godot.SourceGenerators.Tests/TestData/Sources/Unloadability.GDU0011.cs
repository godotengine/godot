using System.Threading;
using Godot;

// Positive: [Tool] class calling ThreadPool.QueueUserWorkItem triggers GDU0011
[Tool]
public class ToolClassWithQueueUserWorkItem
{
    public void QueueWork()
    {
        {|GDU0011:ThreadPool.QueueUserWorkItem(_ => { })|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithQueueUserWorkItem
{
    public void QueueWork()
    {
        ThreadPool.QueueUserWorkItem(_ => { });
    }
}
