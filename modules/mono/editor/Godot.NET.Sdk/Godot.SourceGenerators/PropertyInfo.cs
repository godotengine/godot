namespace Godot.SourceGenerators
{
    internal readonly struct PropertyInfo
    {
        public PropertyInfo(VariantType type, string name, PropertyHint hint,
            string? hintString, PropertyUsageFlags usage, bool exported)
        {
            Type = type;
            Name = name;
            Hint = hint;
            HintString = hintString;
            Usage = usage;
            Exported = exported;
        }

        public VariantType Type { get; }
        public string Name { get; }
        public PropertyHint Hint { get; }
        public string? HintString { get; }
        public PropertyUsageFlags Usage { get; }
        public bool Exported { get; }
    }
}
