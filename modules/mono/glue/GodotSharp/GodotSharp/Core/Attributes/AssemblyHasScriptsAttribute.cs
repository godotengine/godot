using System;

namespace Godot
{
    /// <summary>
    /// An attribute that determines if an assembly has scripts. If so, what types of scripts the assembly has.
    /// </summary>
    [AttributeUsage(AttributeTargets.Assembly)]
    public class AssemblyHasScriptsAttribute : Attribute
    {
        private readonly bool requiresLookup;
        private readonly System.Type[] scriptTypes;

        /// <summary>
        /// Constructs a new AssemblyHasScriptsAttribute instance.
        /// </summary>
        public AssemblyHasScriptsAttribute()
        {
            requiresLookup = true;
        }

        /// <summary>
        /// Constructs a new AssemblyHasScriptsAttribute instance.
        /// </summary>
        /// <param name="scriptTypes">The specified type(s) of scripts.</param>
        public AssemblyHasScriptsAttribute(System.Type[] scriptTypes)
        {
            requiresLookup = false;
            this.scriptTypes = scriptTypes;
        }
    }
}
