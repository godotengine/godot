using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    /// <summary>
    /// The Rid type is used to access the unique integer ID of a resource.
    /// They are opaque, which means they do not grant access to the associated
    /// resource by themselves. They are used by and with the low-level Server
    /// classes such as <see cref="RenderingServer"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Rid
    {
        private readonly ulong _id; // Default is 0

        internal Rid(ulong id)
        {
            _id = id;
        }

        /// <summary>
        /// Constructs a new <see cref="Rid"/> for the given <see cref="GodotObject"/> <paramref name="from"/>.
        /// </summary>
        public Rid(GodotObject from)
            => _id = from is Resource res ? res.GetRid()._id : default;

        /// <summary>
        /// Returns the ID of the referenced resource.
        /// </summary>
        /// <returns>The ID of the referenced resource.</returns>
        public ulong Id => _id;

        /// <summary>
        /// Converts this <see cref="Rid"/> to a string.
        /// </summary>
        /// <returns>A string representation of this Rid.</returns>
        public override string ToString() => $"Rid({Id})";
    }
}
