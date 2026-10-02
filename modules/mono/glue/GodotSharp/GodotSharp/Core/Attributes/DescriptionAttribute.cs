using System;

namespace Godot
{
    /// <summary>
    /// Provides help text in the Inspector for a field or property exported with
    /// <see cref="ExportAttribute"/> or <see cref="ExportToolButtonAttribute"/>.
    /// This attribute does not export the member on its own.
    /// </summary>
    /// <remarks>
    /// The description supports Godot documentation BBCode and line breaks.
    /// Rebuild the project to update the description in the editor.
    /// </remarks>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, Inherited = false)]
    public sealed class DescriptionAttribute : Attribute
    {
        /// <summary>
        /// The help text displayed in the Inspector.
        /// </summary>
        public string Description { get; }

        /// <summary>
        /// Constructs an attribute with the specified Inspector help text.
        /// </summary>
        /// <param name="description">The description of the exported member.</param>
        public DescriptionAttribute(string description)
        {
            Description = description;
        }
    }
}
