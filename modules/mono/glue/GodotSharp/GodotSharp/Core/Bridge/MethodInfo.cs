using System.Collections.Generic;

namespace Godot.Bridge;

#nullable enable

/// <summary>
/// Collection of information regarding a given method.
/// </summary>
public readonly struct MethodInfo
{
    /// <summary>
    /// The name of this method.
    /// </summary>
    public StringName Name { get; init; }
    /// <summary>
    /// The return value of this method.
    /// </summary>
    public PropertyInfo ReturnVal { get; init; }
    /// <summary>
    /// Implementation flags for this method.
    /// </summary>
    public MethodFlags Flags { get; init; }
    /// <summary>
    /// The id of this method.
    /// </summary>
    public int Id { get; init; } = 0;
    /// <summary>
    /// A list of arguments for this method.
    /// </summary>
    public List<PropertyInfo>? Arguments { get; init; }
    /// <summary>
    /// A list of default arguments for this method.
    /// </summary>
    public List<Variant>? DefaultArguments { get; init; }

    /// <summary>
    /// Constructs a new <see cref="MethodInfo"/> with the provided arguments.
    /// </summary>
    /// <param name="name">The method's name.</param>
    /// <param name="returnVal">The method's return value.</param>
    /// <param name="flags">The method's implementation flags.</param>
    /// <param name="arguments">The method's arguments.</param>
    /// <param name="defaultArguments">The method's default arguments.</param>
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
