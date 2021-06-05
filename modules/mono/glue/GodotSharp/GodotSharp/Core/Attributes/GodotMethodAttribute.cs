using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Method)]
    internal class GodotMethodAttribute : Attribute
    {
        private string methodName;

        public string MethodName => methodName;

        public GodotMethodAttribute(string methodName) => this.methodName = methodName;
    }
}
