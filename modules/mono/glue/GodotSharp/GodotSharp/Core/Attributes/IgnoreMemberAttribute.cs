using System;

namespace Godot
{
    /// <summary>
    /// An attribute that excludes a member from registering as a method or property in Godot.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class IgnoreMemberAttribute : Attribute { }
}
