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
        /// Optional usage flags that determine how the property should be handled by the editor.<br/>
        /// Note: Regardless of the usage value, the <see cref="PropertyUsageFlags.ScriptVariable"/> flag is always added,
        /// as with any explicitly declared script variable.
        /// </summary>
        public PropertyUsageFlags Usage { get; }

        /// <summary>
        /// Constructs a new ExportAttribute Instance.
        /// </summary>
        /// <param name="hint">The hint for the exported property.</param>
        /// <param name="hintString">A string that may contain additional metadata for the hint.</param>
        /// <param name="usage">The usage flags for the exported property.</param>
        public ExportAttribute(PropertyHint hint = PropertyHint.None, string hintString = "", PropertyUsageFlags usage = PropertyUsageFlags.Default)
        {
            Hint = hint;
            HintString = hintString;
            Usage = usage;
        }
    }
}
