using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public class ScriptPathAttribute : Attribute
    {
        public string Path { get; private set; }

        public ScriptPathAttribute(string path)
        {
            Path = path;
        }
    }
}
