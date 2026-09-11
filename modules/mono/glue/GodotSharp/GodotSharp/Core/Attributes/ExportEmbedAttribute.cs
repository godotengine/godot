using System;

namespace Godot
{
    /// <summary>
    /// Embeds the properties of annotated member as a properties of the Godot Object.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class ExportEmbedAttribute : Attribute { }
}
