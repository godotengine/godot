using System;

#nullable enable

namespace Godot
{
    /// <summary>
    /// An attribute that determines if an assembly has scripts. If so, what types of scripts the assembly has.
    /// </summary>
    [AttributeUsage(AttributeTargets.Assembly)]
    public class AssemblyHasScriptsAttribute : Attribute
    {
        public bool RequiresLookup { get; }
        public Type[]? ScriptTypes { get; }

        /// <summary>
        /// Constructs a new AssemblyHasScriptsAttribute instance.
        /// </summary>
        public AssemblyHasScriptsAttribute()
        {
            RequiresLookup = true;
            ScriptTypes = null;
        }

        /// <summary>
        /// Constructs a new AssemblyHasScriptsAttribute instance.
        /// </summary>
        /// <param name="scriptTypes">The specified type(s) of scripts.</param>
        public AssemblyHasScriptsAttribute(Type[] scriptTypes)
        {
            RequiresLookup = false;
            ScriptTypes = scriptTypes;
        }
    }
}

#nullable restore
