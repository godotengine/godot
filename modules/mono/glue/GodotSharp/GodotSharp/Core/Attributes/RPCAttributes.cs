using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Method)]
    public class AnyAttribute : Attribute { }

    [AttributeUsage(AttributeTargets.Method)]
    public class AuthorityAttribute : Attribute { }
}
