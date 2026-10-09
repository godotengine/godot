using System;
using Godot.Bridge;

namespace Godot
{
    public partial class CSharpScript
    {
        /// <summary>
        /// Return the <see cref="Type"/> of the C# Script associated to this instance.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The C# Script cannot be found (in case of an invalid pointer).
        /// </exception>
        public Type GetScriptType()
        {
            return ScriptManagerBridge.GetManagedScriptType(GetPtr(this));
        }
    }
}
