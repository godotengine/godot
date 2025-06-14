using System;

#nullable enable

namespace Godot
{
    /// <summary>
    /// Exposes the target class as a global script class to Godot Engine.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class GlobalClassAttribute : Attribute
    {
        /// <summary>
        /// Prevent the current class from appearing in the Create Node dialog.
        /// </summary>
        public bool Hidden { get; init; }
    }
}
