using System;

namespace Godot
{
    /// <summary>
    /// Exports the annotated member as a property of the Godot Object.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportAttribute : Attribute
    {
        /// <summary>
        /// Optional hint that determines how the property should be handled by the editor.
        /// </summary>
        public PropertyHint Hint { get; }

        /// <summary>
        /// Optional string that can contain additional metadata for the <see cref="Hint"/>.
        /// </summary>
        public string HintString { get; }

        /// <summary>
        /// Optional usage flag that determines how the property will get used in the engine.
        /// </summary>
        public PropertyUsageFlags UsageFlags { get; }

        /// <summary>
        /// Constructs a new ExportAttribute Instance.
        /// </summary>
        /// <param name="hint">The hint for the exported property.</param>
        /// <param name="hintString">A string that may contain additional metadata for the hint.</param>
        /// <param name="usageFlags">The usage flags for the exported property.</param>
        public ExportAttribute(PropertyHint hint = PropertyHint.None, string hintString = "",
            PropertyUsageFlags usageFlags = PropertyUsageFlags.Default)
        {
            Hint = hint;
            HintString = hintString;
            UsageFlags = usageFlags;
        }
    }
}
