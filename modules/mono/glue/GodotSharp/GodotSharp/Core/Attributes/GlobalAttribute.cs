using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Class)]
    public class GlobalAttribute : Attribute
    {
        private string name;
        private string iconPath;

        public GlobalAttribute(string name = "", string iconPath = "")
        {
            this.name = name;
            this.iconPath = iconPath;
        }
    }
}
