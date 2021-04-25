using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Assembly)]
    public class AssemblyHasScriptsAttribute : Attribute
    {
        private readonly bool requiresLookup;
        private readonly System.Type[] scriptTypes;

        public AssemblyHasScriptsAttribute()
        {
            requiresLookup = true;
        }

        public AssemblyHasScriptsAttribute(System.Type[] scriptTypes)
        {
            requiresLookup = false;
            this.scriptTypes = scriptTypes;
        }
    }
}
