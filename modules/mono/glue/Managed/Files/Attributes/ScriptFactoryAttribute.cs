using System;

namespace Godot
{
    /// <inheritdoc />
    /// <summary>
    /// This attribute can be used on your project's assembly to tell Godot
    /// about a factory for creating Script instances.
    /// </summary>
    public class ScriptInstanceFactoryAttribute : Attribute
    {

        public Type FactoryType { get; }

        public ScriptInstanceFactoryAttribute(Type type)
        {
            FactoryType = type;
        }

    }
}
