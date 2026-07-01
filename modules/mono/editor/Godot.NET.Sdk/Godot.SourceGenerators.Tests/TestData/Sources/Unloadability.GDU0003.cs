using System;
using System.Threading;
using Godot;

// Positive: [Tool] class calling ThreadPool.RegisterWaitForSingleObject triggers GDU0003
[Tool]
public class ToolClassWithRegisterWait
{
    public void Register()
    {
        var waitHandle = new ManualResetEvent(false);
        {|GDU0003:ThreadPool.RegisterWaitForSingleObject(waitHandle, Callback, null, -1, true)|};
    }

    private void Callback(object state, bool timedOut) { }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithRegisterWait
{
    public void Register()
    {
        var waitHandle = new ManualResetEvent(false);
        ThreadPool.RegisterWaitForSingleObject(waitHandle, Callback, null, -1, true);
    }

    private void Callback(object state, bool timedOut) { }
}
