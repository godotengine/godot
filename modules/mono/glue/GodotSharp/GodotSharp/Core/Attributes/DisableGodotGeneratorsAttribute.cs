using System;

namespace Godot
{
    /// <summary>
    /// An attribute that disables Godot Generators.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public class DisableGodotGeneratorsAttribute : Attribute { }
}
