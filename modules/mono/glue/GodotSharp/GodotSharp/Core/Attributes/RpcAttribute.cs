using System;

namespace Godot
{
    /// <summary>
    /// Attribute that changes the RPC mode for the annotated <c>method</c> to the given <see cref="Mode"/>,
    /// optionally specifying the <see cref="TransferMode"/> and <see cref="TransferChannel"/> (on supported peers).
    /// See <see cref="MultiplayerApi.RpcMode"/> and <see cref="MultiplayerPeer.TransferModeEnum"/>.
    /// By default, methods are not exposed to networking (and RPCs).
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = false)]
    public sealed class RpcAttribute : Attribute
    {
        /// <summary>
        /// RPC mode for the annotated method.
        /// </summary>
        public MultiplayerApi.RpcMode Mode { get; } = MultiplayerApi.RpcMode.Disabled;

        /// <summary>
        /// If the method will also be called locally; otherwise, it is only called remotely.
        /// </summary>
        public bool CallLocal { get; init; } = false;

        /// <summary>
        /// Transfer mode for the annotated method.
        /// </summary>
        public MultiplayerPeer.TransferModeEnum TransferMode { get; init; } = MultiplayerPeer.TransferModeEnum.Reliable;

        /// <summary>
        /// Transfer channel for the annotated mode.
        /// </summary>
        public int TransferChannel { get; init; } = 0;

        /// <summary>
        /// Constructs a <see cref="RpcAttribute"/> instance.
        /// </summary>
        /// <param name="mode">The RPC mode to use.</param>
        public RpcAttribute(MultiplayerApi.RpcMode mode = MultiplayerApi.RpcMode.Authority)
        {
            Mode = mode;
        }
    }
}
