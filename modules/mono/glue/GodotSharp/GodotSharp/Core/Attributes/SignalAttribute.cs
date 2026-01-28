using System;

namespace Godot
{
    /// <summary>
    /// Declares a <see langword="delegate"/> as a signal. This allows any connected
    /// <see cref="Callable"/> (and, by extension, their respective objects) to listen and react
    /// to events, without directly referencing one another.
    /// </summary>
    [AttributeUsage(AttributeTargets.Delegate)]
    public sealed class SignalAttribute : Attribute { }
}
