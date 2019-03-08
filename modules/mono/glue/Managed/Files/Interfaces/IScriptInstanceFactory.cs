using System;

namespace Godot
{
    /// <summary>
    /// Implement this interface to take control of script instance
    /// creation for scripts that are attached to nodes.
    /// Use <see cref="Godot.ScriptInstanceFactoryAttribute"/>
    /// on your assembly to register a factory with Godot.
    /// </summary>
    public interface IScriptInstanceFactory
    {

        void Initialize(Godot.Object uninitializedObject, object[] args);

    }
}
