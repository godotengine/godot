using System;
using System.Runtime.InteropServices;
using Godot;

// Positive: [Tool] class calling GCHandle.Alloc (default = Normal) triggers GDU0002
[Tool]
public class ToolClassWithGCHandle
{
    public void AllocHandle()
    {
        var obj = new object();
        var handle = {|GDU0002:GCHandle.Alloc(obj)|};
    }

    public void AllocPinnedHandle()
    {
        var data = new byte[10];
        var handle = {|GDU0002:GCHandle.Alloc(data, GCHandleType.Pinned)|};
    }
}

// Negative: Weak GCHandles do NOT create strong roots — should NOT trigger
[Tool]
public class ToolClassWithWeakGCHandle
{
    public void AllocWeakHandle()
    {
        var obj = new object();
        var handle = GCHandle.Alloc(obj, GCHandleType.Weak);
    }

    public void AllocWeakTrackResurrectionHandle()
    {
        var obj = new object();
        var handle = GCHandle.Alloc(obj, GCHandleType.WeakTrackResurrection);
    }
}

// Negative: non-Tool class calling GCHandle.Alloc should NOT trigger
public class NonToolClassWithGCHandle
{
    public void AllocHandle()
    {
        var obj = new object();
        var handle = GCHandle.Alloc(obj);
    }
}
