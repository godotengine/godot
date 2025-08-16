using System;

#nullable enable

namespace Godot
{
    /// <summary>
    /// Define a new group for the following exported properties. This helps to organize properties in the Inspector dock.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportGroupAttribute : Attribute
    {
        /// <summary>
        /// Name of the group.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// If provided, the prefix that all properties must have to be considered part of the group.
        /// </summary>
        public string? Prefix { get; }

        /// <summary>
        /// Define a new group for the following exported properties.
        /// </summary>
        /// <param name="name">The name of the group.</param>
        /// <param name="prefix">If provided, the group would make group to only consider properties that have this prefix.</param>
        public ExportGroupAttribute(string name, string prefix = "")
        {
            Name = name;
            Prefix = prefix;
        }
    }
}
