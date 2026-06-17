using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot
{


[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Vector2i : IEquatable<Vector2i>
{
    public enum Axis
    {
        X = 0,
        Y
    }

    public int X;
    public int Y;

    public int this[int index]
    {
        readonly get
        {
            switch (index)
            {
                case 0:
                    return X;
                case 1:
                    return Y;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
        set
        {
            switch (index)
            {
                case 0:
                    X = value;
                    return;
                case 1:
                    Y = value;
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    public readonly void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    public readonly Vector2i Abs()
    {
        return new Vector2i(Mathf.Abs(X), Mathf.Abs(Y));
    }

    public readonly float Aspect()
    {
        return X / (float)Y;
    }

    public readonly Vector2i Clamp(Vector2i min, Vector2i max)
    {
        return new Vector2i
        (
            Mathf.Clamp(X, min.X, max.X),
            Mathf.Clamp(Y, min.Y, max.Y)
        );
    }

    public readonly Vector2i Clamp(int min, int max)
    {
        return new Vector2i
        (
            Mathf.Clamp(X, min, max),
            Mathf.Clamp(Y, min, max)
        );
    }

    public readonly int DistanceSquaredTo(Vector2i to)
    {
        return (to - this).LengthSquared();
    }

    public readonly float DistanceTo(Vector2i to)
    {
        return (to - this).Length();
    }

    public readonly float Length()
    {
        int x2 = X * X;
        int y2 = Y * Y;

        return Mathf.Sqrt(x2 + y2);
    }

    public readonly int LengthSquared()
    {
        int x2 = X * X;
        int y2 = Y * Y;

        return x2 + y2;
    }

    public readonly Vector2i Max(Vector2i with)
    {
        return new Vector2i
        (
            Mathf.Max(X, with.X),
            Mathf.Max(Y, with.Y)
        );
    }

    public readonly Vector2i Max(int with)
    {
        return new Vector2i
        (
            Mathf.Max(X, with),
            Mathf.Max(Y, with)
        );
    }

    public readonly Vector2i Min(Vector2i with)
    {
        return new Vector2i
        (
            Mathf.Min(X, with.X),
            Mathf.Min(Y, with.Y)
        );
    }

    public readonly Vector2i Min(int with)
    {
        return new Vector2i
        (
            Mathf.Min(X, with),
            Mathf.Min(Y, with)
        );
    }

    public readonly Axis MaxAxisIndex()
    {
        return X < Y ? Axis.Y : Axis.X;
    }

    public readonly Axis MinAxisIndex()
    {
        return X < Y ? Axis.X : Axis.Y;
    }

    public readonly Vector2i Sign()
    {
        Vector2i v = this;
        v.X = Mathf.Sign(v.X);
        v.Y = Mathf.Sign(v.Y);
        return v;
    }

    public readonly Vector2i Snapped(Vector2i step)
    {
        return new Vector2i
        (
            (int)Mathf.Snapped((double)X, (double)step.X),
            (int)Mathf.Snapped((double)Y, (double)step.Y)
        );
    }

    public readonly Vector2i Snapped(int step)
    {
        return new Vector2i
        (
            (int)Mathf.Snapped((double)X, (double)step),
            (int)Mathf.Snapped((double)Y, (double)step)
        );
    }

    private static readonly Vector2i _minValue = new Vector2i(int.MinValue, int.MinValue);
    private static readonly Vector2i _maxValue = new Vector2i(int.MaxValue, int.MaxValue);

    private static readonly Vector2i _zero = new Vector2i(0, 0);
    private static readonly Vector2i _one = new Vector2i(1, 1);

    private static readonly Vector2i _up = new Vector2i(0, -1);
    private static readonly Vector2i _down = new Vector2i(0, 1);
    private static readonly Vector2i _right = new Vector2i(1, 0);
    private static readonly Vector2i _left = new Vector2i(-1, 0);

    public static Vector2i MinValue { get { return _minValue; } }
    public static Vector2i MaxValue { get { return _maxValue; } }

    public static Vector2i Zero { get { return _zero; } }
    public static Vector2i One { get { return _one; } }

    public static Vector2i Up { get { return _up; } }
    public static Vector2i Down { get { return _down; } }
    public static Vector2i Right { get { return _right; } }
    public static Vector2i Left { get { return _left; } }

    public Vector2i(int x, int y)
    {
        X = x;
        Y = y;
    }

    public static Vector2i operator +(Vector2i left, Vector2i right)
    {
        left.X += right.X;
        left.Y += right.Y;
        return left;
    }

    public static Vector2i operator -(Vector2i left, Vector2i right)
    {
        left.X -= right.X;
        left.Y -= right.Y;
        return left;
    }

    public static Vector2i operator -(Vector2i vec)
    {
        vec.X = -vec.X;
        vec.Y = -vec.Y;
        return vec;
    }

    public static Vector2i operator *(Vector2i vec, int scale)
    {
        vec.X *= scale;
        vec.Y *= scale;
        return vec;
    }

    public static Vector2i operator *(int scale, Vector2i vec)
    {
        vec.X *= scale;
        vec.Y *= scale;
        return vec;
    }

    public static Vector2i operator *(Vector2i left, Vector2i right)
    {
        left.X *= right.X;
        left.Y *= right.Y;
        return left;
    }

    public static Vector2i operator /(Vector2i vec, int divisor)
    {
        vec.X /= divisor;
        vec.Y /= divisor;
        return vec;
    }

    public static Vector2i operator /(Vector2i vec, Vector2i divisorv)
    {
        vec.X /= divisorv.X;
        vec.Y /= divisorv.Y;
        return vec;
    }

    public static Vector2i operator %(Vector2i vec, int divisor)
    {
        vec.X %= divisor;
        vec.Y %= divisor;
        return vec;
    }

    public static Vector2i operator %(Vector2i vec, Vector2i divisorv)
    {
        vec.X %= divisorv.X;
        vec.Y %= divisorv.Y;
        return vec;
    }

    public static bool operator ==(Vector2i left, Vector2i right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(Vector2i left, Vector2i right)
    {
        return !left.Equals(right);
    }

    public static bool operator <(Vector2i left, Vector2i right)
    {
        if (left.X == right.X)
        {
            return left.Y < right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >(Vector2i left, Vector2i right)
    {
        if (left.X == right.X)
        {
            return left.Y > right.Y;
        }
        return left.X > right.X;
    }

    public static bool operator <=(Vector2i left, Vector2i right)
    {
        if (left.X == right.X)
        {
            return left.Y <= right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >=(Vector2i left, Vector2i right)
    {
        if (left.X == right.X)
        {
            return left.Y >= right.Y;
        }
        return left.X > right.X;
    }

    public static implicit operator Vector2(Vector2i value)
    {
        return new Vector2(value.X, value.Y);
    }

    public static explicit operator Vector2i(Vector2 value)
    {
        return new Vector2i((int)value.X, (int)value.Y);
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Vector2i other && Equals(other);
    }

    public readonly bool Equals(Vector2i other)
    {
        return X == other.X && Y == other.Y;
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(X, Y);
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)})";
    }
}
}