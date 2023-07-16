namespace Godot.Bridge;

#nullable enable

/// <summary>
/// Collection of information regarding a given property.
/// </summary>
public readonly struct PropertyInfo
{
    /// <summary>
    /// The <see cref="Variant"/> type of this property.
    /// </summary>
    public Variant.Type Type { get; init; }
    /// <summary>
    /// The name of this property.
    /// </summary>
    public StringName Name { get; init; }
    /// <summary>
    /// The hint of this property.
    /// </summary>
    public PropertyHint Hint { get; init; }
    /// <summary>
    /// The hint string of this property.
    /// </summary>
    public string HintString { get; init; }
    /// <summary>
    /// The usage flags for this property.
    /// </summary>
    public PropertyUsageFlags Usage { get; init; }
    /// <summary>
    /// The class name of this property.
    /// </summary>
    public StringName? ClassName { get; init; }
    /// <summary>
    /// The export status of this property.
    /// </summary>
    public bool Exported { get; init; }

    /// <summary>
    /// Constructs a new <see cref="PropertyInfo"/> with the provided arguments.
    /// </summary>
    /// <param name="type">The property's type.</param>
    /// <param name="name">The property's name.</param>
    /// <param name="hint">The property's hint.</param>
    /// <param name="hintString">The property's hint string.</param>
    /// <param name="usage">The property's usage flags.</param>
    /// <param name="exported">The property's export state.</param>
    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, bool exported)
        : this(type, name, hint, hintString, usage, className: null, exported) { }

    /// <summary>
    /// Constructs a new <see cref="PropertyInfo"/> with the provided arguments.
    /// </summary>
    /// <param name="type">The property's type.</param>
    /// <param name="name">The property's name.</param>
    /// <param name="hint">The property's hint.</param>
    /// <param name="hintString">The property's hint string.</param>
    /// <param name="usage">The property's usage flags.</param>
    /// <param name="className">The property's class name.</param>
    /// <param name="exported">The property's export state.</param>
    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, StringName? className, bool exported)
    {
        Type = type;
        Name = name;
        Hint = hint;
        HintString = hintString;
        Usage = usage;
        ClassName = className;
        Exported = exported;
    }
}
