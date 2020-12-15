using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Class)]
    public class ScriptClassAttribute : Attribute
    {
        private string name;
        private string iconPath;

        public ScriptClassAttribute(string name = "", string iconPath = "")
        {
            this.name = name;
            this.iconPath = iconPath;
        }
    }
}
