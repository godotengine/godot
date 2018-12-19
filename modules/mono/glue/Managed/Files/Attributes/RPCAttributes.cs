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
    public class PuppetAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class SlaveAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class RemoteSyncAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class MasterSyncAttribute : Attribute {}

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field)]
    public class PuppetSyncAttribute : Attribute {}
}
