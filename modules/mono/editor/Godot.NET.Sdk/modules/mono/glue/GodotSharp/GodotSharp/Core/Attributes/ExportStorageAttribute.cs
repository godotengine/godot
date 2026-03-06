using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportStorageAttribute : Attribute
    {
        public PropertyHint Hint { get; }

        public string HintString { get; }

        public ExportStorageAttribute(PropertyHint hint = PropertyHint.None, string hintString = "")
        {
            Hint = hint;
            HintString = hintString;
        }
    }
}
