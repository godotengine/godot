using System;

namespace Godot
{
    /// <summary>
    /// Constructs a new AnyPeerAttribute instance. Members with the AnyPeerAttribute are given authority over their own player.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class AnyPeerAttribute : Attribute { }

    /// <summary>
    /// Constructs a new AuthorityAttribute instance. Members with the AuthorityAttribute are given authority over the game.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method)]
    public class AuthorityAttribute : Attribute { }
}
