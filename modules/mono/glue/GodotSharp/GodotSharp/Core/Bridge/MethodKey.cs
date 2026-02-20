using System;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    public readonly struct MethodKey : IEquatable<MethodKey>
    {
        private readonly int _hashCode;

#pragma warning disable CA1051 // Do not declare visible instance fields
        public readonly StringName Name;
        public readonly int ArgCount;
#pragma warning restore CA1051 // Do not declare visible instance fields

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public MethodKey(StringName name, int argCount)
        {
            Name = name;
            ArgCount = argCount;

            _hashCode = HashCode.Combine(name, argCount);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(MethodKey other) =>
            _hashCode == other._hashCode && // Quick rejection
            ArgCount == other.ArgCount &&
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
