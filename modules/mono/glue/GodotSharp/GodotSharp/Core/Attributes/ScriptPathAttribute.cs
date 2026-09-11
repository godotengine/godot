using System;
using JetBrains.Annotations;

#nullable enable

namespace Godot
{
    /// <summary>
    /// An attribute that contains the path to the object's script.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, Inherited = false)]
    [PublicAPI("ABI compatibility with legacy code.")]
    public sealed class ScriptPathAttribute : Attribute
    {
        /// <summary>
        /// File path to the script.
        /// </summary>
        public string Path { get; }

        /// <summary>
        /// Constructs a new ScriptPathAttribute instance.
        /// </summary>
        /// <param name="path">The file path to the script</param>
        public ScriptPathAttribute(string path)
        {
            Path = path;
        }
    }
}
