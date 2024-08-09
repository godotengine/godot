using System;

namespace Godot
{
    /// <summary>
    /// Define a resource to populate with a GetNode before _Ready.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public sealed class GetNodeAttribute : Attribute
    {
        /// <summary>
        /// Path of the resource in the scene.
        /// </summary>
        public string Path { get; }

        /// <summary>
        /// Constructs a new GetNodeAttribute Instance.
        /// </summary>
        /// <param name="path">Path of the resource in the scene.</param>
        public GetNodeAttribute(string path)
        {
            Path = path;
        }
    }
}
