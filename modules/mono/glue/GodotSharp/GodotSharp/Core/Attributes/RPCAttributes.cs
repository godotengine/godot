using System;

namespace Godot
{
    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute remotely.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class RemoteAttribute : Attribute { }

    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute remotely,
    /// but will also execute locally (do a normal method call).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class RemoteSyncAttribute : Attribute { }

    /// <summary>
    /// Same as <see cref="RemoteSyncAttribute"/>.
    /// </summary>
    [Obsolete("Use the RemoteSync attribute instead.")]
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class SyncAttribute : Attribute { }

    /// <summary>
    /// Same as <see cref="PuppetAttribute"/>.
    /// </summary>
    [Obsolete("Use the Puppet attribute instead.")]
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class SlaveAttribute : Attribute { }

    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute only
    /// on the peers that are not set as master of the node.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class PuppetAttribute : Attribute { }

    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute only
    /// on the peers that are not set as master of the node but will also execute locally (do a normal method call).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class PuppetSyncAttribute : Attribute { }

    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute only
    /// on the peer that is set as master of the node.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class MasterAttribute : Attribute { }

    /// <summary>
    /// RPC calls to methods annotated with this attribute go via the network and execute only
    /// on the peer that is set as master of the node but will also execute locally (do a normal method call).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Field | AttributeTargets.Property)]
    public class MasterSyncAttribute : Attribute { }
}
