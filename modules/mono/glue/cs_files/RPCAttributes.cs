using System;

namespace Godot
{
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class RemoteAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class SyncAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class MasterAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class SlaveAttribute : Attribute {}
}
