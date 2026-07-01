using System;
using System.ComponentModel;
using Godot;

// Positive: [Tool] class calling TypeDescriptor methods with a user-defined type triggers GDU0006
[Tool]
public class ToolClassWithTypeDescriptor
{
    public void ModifyTypeDescriptor()
    {
        {|GDU0006:TypeDescriptor.Refresh(typeof(ToolClassWithTypeDescriptor))|};
    }
}

// Negative: non-Tool class should NOT trigger
public class NonToolClassWithTypeDescriptor
{
    public void ModifyTypeDescriptor()
    {
        TypeDescriptor.Refresh(typeof(NonToolClassWithTypeDescriptor));
    }
}
