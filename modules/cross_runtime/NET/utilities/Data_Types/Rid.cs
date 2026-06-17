using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
[StructLayout(LayoutKind.Sequential)]
public readonly struct RID : IEquatable<RID>
{
    private readonly ulong _id;

    internal RID(ulong id)
    {
        _id = id;
    }

    public RID(GodotObject from)
    {
        _id = from != null ? from.Id : default;
    }

    public ulong Id => _id;

    public bool IsValid => _id != 0;

    public static bool operator ==(RID left, RID right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(RID left, RID right)
    {
        return !left.Equals(right);
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is RID other && Equals(other);
    }

    public readonly bool Equals(RID other)
    {
        return _id == other._id;
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(_id);
    }

    public override readonly string ToString()
    {
        return $"RID({Id.ToString(null, CultureInfo.InvariantCulture)})";
    }
}

}