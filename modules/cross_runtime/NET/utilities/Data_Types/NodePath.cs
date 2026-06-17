using System;
using System.Diagnostics.CodeAnalysis;

#nullable enable
namespace Godot
{
/// <summary>
/// A pre-parsed relative or absolute path in a scene tree.
/// </summary>
public readonly struct NodePath : IEquatable<NodePath>
{
    private readonly string _path;

    public NodePath()
    {
        _path = string.Empty;
    }

    public NodePath(string path)
    {
        _path = path ?? string.Empty;
    }

    public static implicit operator NodePath(string from) => new NodePath(from);

    public static implicit operator string?(NodePath from)
    {
        return from.IsEmpty ? string.Empty : from._path;
    }

    public override readonly string ToString()
    {
        return _path ?? string.Empty;
    }

    public readonly NodePath GetAsPropertyPath()
    {
        if (IsEmpty)
            return new NodePath(string.Empty);

        if (_path.StartsWith(":", StringComparison.Ordinal))
            return this;

        int colonIndex = _path.IndexOf(':');
        if (colonIndex < 0)
            return new NodePath(":" + _path);

        return new NodePath(":" + _path);
    }

    public readonly string GetConcatenatedNames()
    {
        if (IsEmpty)
            return string.Empty;

        int colonIndex = _path.IndexOf(':');
        if (colonIndex < 0)
            return _path;

        return _path.Substring(0, colonIndex);
    }

    public readonly string GetConcatenatedSubNames()
    {
        if (IsEmpty)
            return string.Empty;

        int colonIndex = _path.IndexOf(':');
        if (colonIndex < 0)
            return string.Empty;

        return _path.Substring(colonIndex + 1);
    }

    public readonly string GetName(int idx)
    {
        string namesPart = GetConcatenatedNames();
        if (string.IsNullOrEmpty(namesPart))
            throw new ArgumentOutOfRangeException(nameof(idx));

        string[] names = namesPart.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (idx < 0 || idx >= names.Length)
            throw new ArgumentOutOfRangeException(nameof(idx));

        return names[idx];
    }

    public readonly int GetNameCount()
    {
        string namesPart = GetConcatenatedNames();
        if (string.IsNullOrEmpty(namesPart))
            return 0;

        return namesPart.Split('/', StringSplitOptions.RemoveEmptyEntries).Length;
    }

    public readonly string GetSubName(int idx)
    {
        string subNamesPart = GetConcatenatedSubNames();
        if (string.IsNullOrEmpty(subNamesPart))
            throw new ArgumentOutOfRangeException(nameof(idx));

        string[] subNames = subNamesPart.Split(':', StringSplitOptions.RemoveEmptyEntries);
        if (idx < 0 || idx >= subNames.Length)
            throw new ArgumentOutOfRangeException(nameof(idx));

        return subNames[idx];
    }

    public readonly int GetSubNameCount()
    {
        string subNamesPart = GetConcatenatedSubNames();
        if (string.IsNullOrEmpty(subNamesPart))
            return 0;

        return subNamesPart.Split(':', StringSplitOptions.RemoveEmptyEntries).Length;
    }

    public readonly bool IsAbsolute()
    {
        return !string.IsNullOrEmpty(_path) && _path[0] == '/';
    }

    public readonly bool IsEmpty => string.IsNullOrEmpty(_path);

    public readonly bool Equals(NodePath other)
    {
        return string.Equals(_path, other._path, StringComparison.Ordinal);
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is NodePath other && Equals(other);
    }

    public override readonly int GetHashCode()
    {
        return StringComparer.Ordinal.GetHashCode(_path ?? string.Empty);
    }

    public static bool operator ==(NodePath left, NodePath right) => left.Equals(right);
    public static bool operator !=(NodePath left, NodePath right) => !left.Equals(right);
}
}