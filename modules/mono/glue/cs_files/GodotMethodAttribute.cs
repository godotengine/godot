using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Method, Inherited = true)]
    internal class GodotMethodAttribute : Attribute
    {
        private string methodName;

        public string MethodName { get { return methodName; } }

        public GodotMethodAttribute(string methodName)
        {
            this.methodName = methodName;
        }
    }
}
