using System.Collections.Generic;

namespace Godot.Bridge;

#nullable enable

public readonly struct MethodInfo
{
    public StringName Name { get; init; }
    public PropertyInfo ReturnVal { get; init; }
    public MethodFlags Flags { get; init; }
    public int Id { get; init; } = 0;
    public List<PropertyInfo>? Arguments { get; init; }
    public List<Variant>? DefaultArguments { get; init; }

    public MethodInfo(StringName name, PropertyInfo returnVal, MethodFlags flags,
        List<PropertyInfo>? arguments, List<Variant>? defaultArguments)
    {
        Name = name;
        ReturnVal = returnVal;
        Flags = flags;
        Arguments = arguments;
        DefaultArguments = defaultArguments;
    }
}
