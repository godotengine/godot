using System;
using Godot;

// Positive: [Tool] class subscribing to external static event triggers GDU0001
[Tool]
public class ToolClassWithStaticEvent
{
    public void Subscribe()
    {
        {|GDU0001:Console.CancelKeyPress += OnCancel|};
    }

    private void OnCancel(object sender, ConsoleCancelEventArgs e) { }
}

// Negative: non-Tool class subscribing to external static event should NOT trigger
public class NonToolClassWithStaticEvent
{
    public void Subscribe()
    {
        Console.CancelKeyPress += OnCancel;
    }

    private void OnCancel(object sender, ConsoleCancelEventArgs e) { }
}

// Negative: [Tool] class unsubscribing (-=) should NOT trigger
[Tool]
public class ToolClassUnsubscribing
{
    public void Unsubscribe()
    {
        Console.CancelKeyPress -= OnCancel;
    }

    private void OnCancel(object sender, ConsoleCancelEventArgs e) { }
}
