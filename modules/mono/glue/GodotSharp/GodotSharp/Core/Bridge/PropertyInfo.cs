namespace Godot.Bridge;

#nullable enable

public readonly struct PropertyInfo
{
    public Variant.Type Type { get; init; }
    public StringName Name { get; init; }
    public PropertyHint Hint { get; init; }
    public string HintString { get; init; }
    public PropertyUsageFlags Usage { get; init; }
    public StringName? ClassName { get; init; }
    public bool Exported { get; init; }
    public string DisplayName { get; init; }

    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, bool exported)
        : this(type, name, hint, hintString, usage, exported, displayName: "") { }

    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, bool exported, string displayName = "")
        : this(type, name, hint, hintString, usage, className: null, exported, displayName) { }

    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, StringName? className, bool exported)
        : this(type, name, hint, hintString, usage, className, exported, displayName: "") { }

    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, StringName? className, bool exported, string displayName = "")
    {
        Type = type;
        Name = name;
        Hint = hint;
        HintString = hintString;
        Usage = usage;
        ClassName = className;
        Exported = exported;
        DisplayName = displayName;
    }
}
