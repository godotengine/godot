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

        /// <summary>
        /// Constructs a new ExportAttribute Instance. Allows properties to expose their data type as a non-C# Resource class in Godot.
        /// </summary>
        /// <param name="className">The name of a class by which to export the targeted property.</param>
        public ExportAttribute(string className)
        {
            // Because `ScriptServer` is not exposed to the scripting API, there is no proper way of validating the passed in className.
            this.hint = PropertyHint.ResourceType;
            this.hintString = className;
        }
    }
}
