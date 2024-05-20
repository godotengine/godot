namespace Godot.SourceGenerators
{
    internal readonly struct PropertyInfo
    {
        public PropertyInfo(VariantType type, string name, PropertyHint hint,
            string? hintString, PropertyUsageFlags usage, bool exported)
            : this(type, name, hint, hintString, usage, className: null, exported) { }

        public PropertyInfo(VariantType type, string name, PropertyHint hint,
            string? hintString, PropertyUsageFlags usage, string? className, bool exported)
        {
            Type = type;
            Name = name;
            Hint = hint;
            HintString = hintString;
            Usage = usage;
            ClassName = className;
            Exported = exported;
        }

        public VariantType Type { get; }
        public string Name { get; }
        public PropertyHint Hint { get; }
        public string? HintString { get; }
        public PropertyUsageFlags Usage { get; }
        public string? ClassName { get; }
        public bool Exported { get; }
    }
}
