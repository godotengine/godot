using System;

namespace Godot
{
    /// <summary>
    /// An attribute that contains the path to the object's script.
    /// </summary>
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ScriptPathAttribute : Attribute
    {
        /// <summary>
        /// File path to the script (res:// or csharp:// path used by the engine).
        /// </summary>
        public string Path { get; }

        /// <summary>
        /// Absolute path to the original source file on disk.
        /// Null for res:// scripts and NuGet packages.
        /// </summary>
        public string SourceFile { get; set; }

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
