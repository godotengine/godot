using System;

namespace Godot
{
    /// <summary>
    /// An attribute for a method.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    internal class GodotMethodAttribute : Attribute
    {
        private string methodName;

        public string MethodName { get { return methodName; } }

        /// <summary>
        /// Constructs a new GodotMethodAttribute instance.
        /// </summary>
        /// <param name="methodName">The name of the method.</param>
        public GodotMethodAttribute(string methodName)
        {
            this.methodName = methodName;
        }
    }
}
