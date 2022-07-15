using System;

namespace Godot
{
    /// <summary>
    /// An attribute that contains the path to the object's script.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public class ScriptPathAttribute : Attribute
    {
        private string path;

        /// <summary>
        /// Constructs a new ScriptPathAttribute instance.
        /// </summary>
        /// <param name="path">The file path to the script</param>
        public ScriptPathAttribute(string path)
        {
            this.path = path;
        }
    }
}
