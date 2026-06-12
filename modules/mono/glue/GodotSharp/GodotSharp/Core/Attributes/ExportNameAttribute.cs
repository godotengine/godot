using System;

namespace Godot
{
    /// <summary>
    /// Overrides the display name used by the editor inspector for the annotated exported member.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportNameAttribute : Attribute
    {
        /// <summary>
        /// The display name to show in the editor inspector.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Overrides the display name used by the editor inspector for the annotated exported member.
        /// </summary>
        /// <param name="name">The display name to show in the editor inspector.</param>
        public ExportNameAttribute(string name)
        {
            Name = name;
        }
    }
}
