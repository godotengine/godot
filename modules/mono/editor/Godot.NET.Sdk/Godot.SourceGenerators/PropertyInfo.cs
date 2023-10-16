using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators
{
    internal readonly struct PropertyInfo
    {
        public PropertyInfo(VariantType? variantType, ITypeSymbol? propertyType, string name, PropertyHint hint,
            string? hintString, PropertyUsageFlags usage, bool exported)
            : this(variantType, propertyType, name, hint, hintString, usage, className: null, exported) { }

        public PropertyInfo(VariantType? variantType, ITypeSymbol? propertyType, string name, PropertyHint hint,
            string? hintString, PropertyUsageFlags usage, string? className, bool exported)
        {
            VariantType = variantType;
            PropertyType = propertyType;
            Name = name;
            Hint = hint;
            HintString = hintString;
            Usage = usage;
            ClassName = className;
            Exported = exported;
        }

        public VariantType? VariantType { get; }
        public ITypeSymbol? PropertyType { get; }
        public string Name { get; }
        public PropertyHint Hint { get; }
        public string? HintString { get; }
        public PropertyUsageFlags Usage { get; }
        public string? ClassName { get; }
        public bool Exported { get; }
    }
}
