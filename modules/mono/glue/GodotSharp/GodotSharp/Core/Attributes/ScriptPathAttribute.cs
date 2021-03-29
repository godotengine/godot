using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public class ScriptPathAttribute : Attribute
    {
        private string path;

        public ScriptPathAttribute(string path)
        {
            this.path = path;
        }
    }
}
