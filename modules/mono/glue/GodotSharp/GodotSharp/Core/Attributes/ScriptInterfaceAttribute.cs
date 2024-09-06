using System;

namespace Godot
{
    /// <summary>
    /// An attribute that contains the fully qualified name of the interface the script implements.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ScriptInterfaceAttribute : Attribute
    {
        /// <summary>
        /// Fully qualified name of the interface the script implements.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// Constructs a new InterfaceAttribute instance.
        /// </summary>
        /// <param name="name">Fully qualified name of the interface the script implements</param>
        public ScriptInterfaceAttribute(string name)
        {
            Name = name;
        }
    }
}
