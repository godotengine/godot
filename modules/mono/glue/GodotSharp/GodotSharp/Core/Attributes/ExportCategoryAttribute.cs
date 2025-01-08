using System;

namespace Godot
{
    /// <summary>
    /// Define a new category for the following exported properties. This helps to organize properties in the Inspector dock.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportCategoryAttribute : Attribute
    {
        /// <summary>
        /// Name of the category.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Define a new category for the following exported properties.
        /// </summary>
        /// <param name="name">The name of the category.</param>
        public ExportCategoryAttribute(string name)
        {
            Name = name;
        }
    }
}
