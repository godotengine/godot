using System;

namespace Godot
{
    /// <summary>
    /// Specifies a custom icon for representing this class in the Godot Editor.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class IconAttribute : Attribute
    {
        /// <summary>
        /// File path to a custom icon for representing this class in the Godot Editor.
        /// </summary>
        public string Path { get; }

        /// <summary>
        /// Specify the custom icon that represents the class.
        /// </summary>
        /// <param name="path">File path to the custom icon.</param>
        public IconAttribute(string path)
        {
            Path = path;
        }
    }
}
