using System;

namespace Godot
{
    /// <summary>
    /// Exposes the target class as a global script class to Godot Engine.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class GlobalAttribute : Attribute
    {
        private string name;
        private string iconPath;

        /// <summary>
        /// Constructs a new GlobalAttribute Instance.
        /// </summary>
        /// <param name="name">The name under which to register the targeted class. Uses its typename by default.</param>
        /// <param name="iconPath">An optional file path to a custom icon for representing this class in the Godot Editor.</param>
        public GlobalAttribute(string name = "", string iconPath = "")
        {
            this.name = name ?? "";
            this.iconPath = iconPath ?? "";
        }
    }
}
