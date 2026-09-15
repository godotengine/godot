namespace Godot.Bridge;

#nullable enable

public readonly struct SignalKey
{
    public StringName Name { get; init; }
    public int ArgumentCount { get; init; } = 0;

    public SignalKey(StringName name, int argumentCount)
    {
        Name = name;
        ArgumentCount = argumentCount;
    }
}
