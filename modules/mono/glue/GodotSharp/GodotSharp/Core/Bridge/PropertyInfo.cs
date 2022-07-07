namespace Godot.Bridge;

#nullable enable

public readonly struct PropertyInfo
{
    public Variant.Type Type { get; init; }
    public StringName Name { get; init; }
    public PropertyHint Hint { get; init; }
    public string HintString { get; init; }
    public PropertyUsageFlags Usage { get; init; }
    public bool Exported { get; init; }

    public PropertyInfo(Variant.Type type, StringName name, PropertyHint hint, string hintString,
        PropertyUsageFlags usage, bool exported)
    {
        Type = type;
        Name = name;
        Hint = hint;
        HintString = hintString;
        Usage = usage;
        Exported = exported;
    }
}
