using System;

namespace Godot
{
    /// <summary>
    /// Attribute that marks the current script as a tool script, allowing it
    /// to be loaded and executed by the editor.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class ToolAttribute : Attribute { }
}
