using System;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    internal readonly struct MethodKey : IEquatable<MethodKey>
    {
        public readonly IntPtr Name;
        public readonly int Argc;
        private readonly int _hashCode;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MethodKey(IntPtr name, int argc)
        {
            Name = name;
            Argc = argc;

            _hashCode = HashCode.Combine(name, argc);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(MethodKey other) =>
            _hashCode == other._hashCode && // Quick rejection
            Argc == other.Argc &&
            Name == other.Name;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override int GetHashCode() => _hashCode;

        public override bool Equals(object obj) => obj is MethodKey mk && Equals(mk);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator ==(MethodKey left, MethodKey right) => left.Equals(right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool operator !=(MethodKey left, MethodKey right) => !left.Equals(right);
    }
}
