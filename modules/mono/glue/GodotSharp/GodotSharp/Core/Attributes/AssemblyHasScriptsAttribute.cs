using System;

#nullable enable

namespace Godot
{
    [AttributeUsage(AttributeTargets.Assembly)]
    public class AssemblyHasScriptsAttribute : Attribute
    {
        public bool RequiresLookup { get; private set; }
        public Type[]? ScriptTypes { get; private set; }

        public AssemblyHasScriptsAttribute()
        {
            RequiresLookup = true;
            ScriptTypes = null;
        }

        public AssemblyHasScriptsAttribute(Type[] scriptTypes)
        {
            RequiresLookup = false;
            ScriptTypes = scriptTypes;
        }
    }
}

#nullable restore
