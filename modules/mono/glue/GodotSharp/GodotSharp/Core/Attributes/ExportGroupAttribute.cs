using System;

namespace Godot
{
    /// <summary>
    /// Define a new group for the following exported properties. This helps to organize properties in the Inspector dock.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportGroupAttribute : Attribute
    {
        private string name;
        private string prefix;

        /// <summary>
        /// Define a new group for the following exported properties.
        /// </summary>
        /// <param name="name">The name of the group.</param>
        /// <param name="prefix">If provided, the group would make group to only consider properties that have this prefix.</param>
        public ExportGroupAttribute(string name, string prefix = "")
        {
            this.name = name;
            this.prefix = prefix;
        }
    }
}
