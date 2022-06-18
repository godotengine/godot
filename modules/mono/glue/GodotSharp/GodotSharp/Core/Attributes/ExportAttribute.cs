using System;

namespace Godot
{
    /// <summary>
    /// An attribute used to export objects.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class ExportAttribute : Attribute
    {
        private PropertyHint hint;
        private string hintString;

        /// <summary>
        /// Constructs a new ExportAttribute Instance.
        /// </summary>
        /// <param name="hint">A hint to the exported object.</param>
        /// <param name="hintString">A string representing the exported object.</param>
        public ExportAttribute(PropertyHint hint = PropertyHint.None, string hintString = "")
        {
            this.hint = hint;
            this.hintString = hintString;
        }
    }
}
