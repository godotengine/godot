using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot
{
    /// <summary>
    /// The RID type is used to access a low-level resource by its unique ID.
    /// RIDs are opaque, which means they do not grant access to the resource
    /// by themselves. They are used by the low-level server classes, such as
    /// <see cref="DisplayServer"/>, <see cref="RenderingServer"/>,
    /// <see cref="TextServer"/>, etc.
    ///
    /// A low-level resource may correspond to a high-level <see cref="Resource"/>,
    /// such as <see cref="Texture"/> or <see cref="Mesh"/>
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct Rid : IEquatable<Rid>
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
        /// Returns the ID of the referenced low-level resource.
        /// </summary>
        /// <returns>The ID of the referenced resource.</returns>
        public ulong Id => _id;

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rid"/> is not <c>0</c>.
        /// </summary>
        /// <returns>Whether or not the ID is valid.</returns>
        public bool IsValid => _id != 0;

        /// <summary>
        /// Returns <see langword="true"/> if both <see cref="Rid"/>s are equal,
        /// which means they both refer to the same low-level resource.
        /// </summary>
        /// <param name="left">The left RID.</param>
        /// <param name="right">The right RID.</param>
        /// <returns>Whether or not the RIDs are equal.</returns>
        public static bool operator ==(Rid left, Rid right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rid"/>s are not equal.
        /// </summary>
        /// <param name="left">The left RID.</param>
        /// <param name="right">The right RID.</param>
        /// <returns>Whether or not the RIDs are equal.</returns>
        public static bool operator !=(Rid left, Rid right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this RID and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the color and the other object are equal.</returns>
        public override readonly bool Equals([NotNullWhen(true)] object? obj)
        {
            return obj is Rid other && Equals(other);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the RIDs are equal.
        /// </summary>
        /// <param name="other">The other RID.</param>
        /// <returns>Whether or not the RIDs are equal.</returns>
        public readonly bool Equals(Rid other)
        {
            return _id == other.Id;
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Rid"/>.
        /// </summary>
        /// <returns>A hash code for this RID.</returns>
        public override readonly int GetHashCode()
        {
            return HashCode.Combine(_id);
        }

        /// <summary>
        /// Converts this <see cref="Rid"/> to a string.
        /// </summary>
        /// <returns>A string representation of this Rid.</returns>
        public override readonly string ToString() => $"RID({Id.ToString(null, CultureInfo.InvariantCulture)})";
    }
}
