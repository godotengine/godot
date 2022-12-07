using System;

#nullable enable

namespace Godot
{
    /// <summary>
    /// Define a new subgroup for the following exported properties. This helps to organize properties in the Inspector dock.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportSubgroupAttribute : Attribute
    {
        /// <summary>
        /// Name of the subgroup.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// If provided, the prefix that all properties must have to be considered part of the subgroup.
        /// </summary>
        public string? Prefix { get; }

        /// <summary>
        /// Define a new subgroup for the following exported properties. This helps to organize properties in the Inspector dock.
        /// </summary>
        /// <param name="name">The name of the subgroup.</param>
        /// <param name="prefix">If provided, the subgroup would make group to only consider properties that have this prefix.</param>
        public ExportSubgroupAttribute(string name, string prefix = "")
        {
            Name = name;
            Prefix = prefix;
        }
    }
}
