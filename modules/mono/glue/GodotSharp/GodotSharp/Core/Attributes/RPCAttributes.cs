using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Method)]
    public class RemoteAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method)]
    public class MasterAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method)]
    public class PuppetAttribute : Attribute {}
}
