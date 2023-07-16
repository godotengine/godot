using System;

namespace Godot
{
    /// <summary>
    /// Attribute that marks the current delegate as a signal, allowing it to
    /// handle engine callbacks.
    /// </summary>
    [AttributeUsage(AttributeTargets.Delegate)]
    public sealed class SignalAttribute : Attribute { }
}
