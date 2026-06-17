using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Vector4i : IEquatable<Vector4i>
{
    public enum Axis
    {
        X = 0,
        Y,
        Z,
        W
    }

    public int X;
    public int Y;
    public int Z;
    public int W;

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
                case 2:
                    return Z;
                case 3:
                    return W;
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
                case 2:
                    Z = value;
                    return;
                case 3:
                    W = value;
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(index));
            }
        }
    }

    public readonly void Deconstruct(out int x, out int y, out int z, out int w)
    {
        x = X;
        y = Y;
        z = Z;
        w = W;
    }

    public readonly Vector4i Abs()
    {
        return new Vector4i(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z), Mathf.Abs(W));
    }

    public readonly Vector4i Clamp(Vector4i min, Vector4i max)
    {
        return new Vector4i
        (
            Mathf.Clamp(X, min.X, max.X),
            Mathf.Clamp(Y, min.Y, max.Y),
            Mathf.Clamp(Z, min.Z, max.Z),
            Mathf.Clamp(W, min.W, max.W)
        );
    }

    public readonly Vector4i Clamp(int min, int max)
    {
        return new Vector4i
        (
            Mathf.Clamp(X, min, max),
            Mathf.Clamp(Y, min, max),
            Mathf.Clamp(Z, min, max),
            Mathf.Clamp(W, min, max)
        );
    }

    public readonly int DistanceSquaredTo(Vector4i to)
    {
        return (to - this).LengthSquared();
    }

    public readonly float DistanceTo(Vector4i to)
    {
        return (to - this).Length();
    }

    public readonly float Length()
    {
        int x2 = X * X;
        int y2 = Y * Y;
        int z2 = Z * Z;
        int w2 = W * W;

        return Mathf.Sqrt(x2 + y2 + z2 + w2);
    }

    public readonly int LengthSquared()
    {
        int x2 = X * X;
        int y2 = Y * Y;
        int z2 = Z * Z;
        int w2 = W * W;

        return x2 + y2 + z2 + w2;
    }

    public readonly Vector4i Max(Vector4i with)
    {
        return new Vector4i
        (
            Mathf.Max(X, with.X),
            Mathf.Max(Y, with.Y),
            Mathf.Max(Z, with.Z),
            Mathf.Max(W, with.W)
        );
    }

    public readonly Vector4i Max(int with)
    {
        return new Vector4i
        (
            Mathf.Max(X, with),
            Mathf.Max(Y, with),
            Mathf.Max(Z, with),
            Mathf.Max(W, with)
        );
    }

    public readonly Vector4i Min(Vector4i with)
    {
        return new Vector4i
        (
            Mathf.Min(X, with.X),
            Mathf.Min(Y, with.Y),
            Mathf.Min(Z, with.Z),
            Mathf.Min(W, with.W)
        );
    }

    public readonly Vector4i Min(int with)
    {
        return new Vector4i
        (
            Mathf.Min(X, with),
            Mathf.Min(Y, with),
            Mathf.Min(Z, with),
            Mathf.Min(W, with)
        );
    }

    public readonly Axis MaxAxisIndex()
    {
        int max_index = 0;
        int max_value = X;
        for (int i = 1; i < 4; i++)
        {
            if (this[i] > max_value)
            {
                max_index = i;
                max_value = this[i];
            }
        }
        return (Axis)max_index;
    }

    public readonly Axis MinAxisIndex()
    {
        int min_index = 0;
        int min_value = X;
        for (int i = 1; i < 4; i++)
        {
            if (this[i] <= min_value)
            {
                min_index = i;
                min_value = this[i];
            }
        }
        return (Axis)min_index;
    }

    public readonly Vector4i Sign()
    {
        return new Vector4i(Mathf.Sign(X), Mathf.Sign(Y), Mathf.Sign(Z), Mathf.Sign(W));
    }

    public readonly Vector4i Snapped(Vector4i step)
    {
        return new Vector4i(
            (int)Mathf.Snapped((double)X, (double)step.X),
            (int)Mathf.Snapped((double)Y, (double)step.Y),
            (int)Mathf.Snapped((double)Z, (double)step.Z),
            (int)Mathf.Snapped((double)W, (double)step.W)
        );
    }

    public readonly Vector4i Snapped(int step)
    {
        return new Vector4i(
            (int)Mathf.Snapped((double)X, (double)step),
            (int)Mathf.Snapped((double)Y, (double)step),
            (int)Mathf.Snapped((double)Z, (double)step),
            (int)Mathf.Snapped((double)W, (double)step)
        );
    }

    private static readonly Vector4i _minValue = new Vector4i(int.MinValue, int.MinValue, int.MinValue, int.MinValue);
    private static readonly Vector4i _maxValue = new Vector4i(int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue);

    private static readonly Vector4i _zero = new Vector4i(0, 0, 0, 0);
    private static readonly Vector4i _one = new Vector4i(1, 1, 1, 1);

    public static Vector4i MinValue { get { return _minValue; } }
    public static Vector4i MaxValue { get { return _maxValue; } }

    public static Vector4i Zero { get { return _zero; } }
    public static Vector4i One { get { return _one; } }

    public Vector4i(int x, int y, int z, int w)
    {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public static Vector4i operator +(Vector4i left, Vector4i right)
    {
        left.X += right.X;
        left.Y += right.Y;
        left.Z += right.Z;
        left.W += right.W;
        return left;
    }

    public static Vector4i operator -(Vector4i left, Vector4i right)
    {
        left.X -= right.X;
        left.Y -= right.Y;
        left.Z -= right.Z;
        left.W -= right.W;
        return left;
    }

    public static Vector4i operator -(Vector4i vec)
    {
        vec.X = -vec.X;
        vec.Y = -vec.Y;
        vec.Z = -vec.Z;
        vec.W = -vec.W;
        return vec;
    }

    public static Vector4i operator *(Vector4i vec, int scale)
    {
        vec.X *= scale;
        vec.Y *= scale;
        vec.Z *= scale;
        vec.W *= scale;
        return vec;
    }

    public static Vector4i operator *(int scale, Vector4i vec)
    {
        vec.X *= scale;
        vec.Y *= scale;
        vec.Z *= scale;
        vec.W *= scale;
        return vec;
    }

    public static Vector4i operator *(Vector4i left, Vector4i right)
    {
        left.X *= right.X;
        left.Y *= right.Y;
        left.Z *= right.Z;
        left.W *= right.W;
        return left;
    }

    public static Vector4i operator /(Vector4i vec, int divisor)
    {
        vec.X /= divisor;
        vec.Y /= divisor;
        vec.Z /= divisor;
        vec.W /= divisor;
        return vec;
    }

    public static Vector4i operator /(Vector4i vec, Vector4i divisorv)
    {
        vec.X /= divisorv.X;
        vec.Y /= divisorv.Y;
        vec.Z /= divisorv.Z;
        vec.W /= divisorv.W;
        return vec;
    }

    public static Vector4i operator %(Vector4i vec, int divisor)
    {
        vec.X %= divisor;
        vec.Y %= divisor;
        vec.Z %= divisor;
        vec.W %= divisor;
        return vec;
    }

    public static Vector4i operator %(Vector4i vec, Vector4i divisorv)
    {
        vec.X %= divisorv.X;
        vec.Y %= divisorv.Y;
        vec.Z %= divisorv.Z;
        vec.W %= divisorv.W;
        return vec;
    }

    public static bool operator ==(Vector4i left, Vector4i right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(Vector4i left, Vector4i right)
    {
        return !left.Equals(right);
    }

    public static bool operator <(Vector4i left, Vector4i right)
    {
        if (left.X == right.X)
        {
            if (left.Y == right.Y)
            {
                if (left.Z == right.Z)
                {
                    return left.W < right.W;
                }
                return left.Z < right.Z;
            }
            return left.Y < right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >(Vector4i left, Vector4i right)
    {
        if (left.X == right.X)
        {
            if (left.Y == right.Y)
            {
                if (left.Z == right.Z)
                {
                    return left.W > right.W;
                }
                return left.Z > right.Z;
            }
            return left.Y > right.Y;
        }
        return left.X > right.X;
    }

    public static bool operator <=(Vector4i left, Vector4i right)
    {
        if (left.X == right.X)
        {
            if (left.Y == right.Y)
            {
                if (left.Z == right.Z)
                {
                    return left.W <= right.W;
                }
                return left.Z < right.Z;
            }
            return left.Y < right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >=(Vector4i left, Vector4i right)
    {
        if (left.X == right.X)
        {
            if (left.Y == right.Y)
            {
                if (left.Z == right.Z)
                {
                    return left.W >= right.W;
                }
                return left.Z > right.Z;
            }
            return left.Y > right.Y;
        }
        return left.X > right.X;
    }

    public static implicit operator Vector4(Vector4i value)
    {
        return new Vector4(value.X, value.Y, value.Z, value.W);
    }

    public static explicit operator Vector4i(Vector4 value)
    {
        return new Vector4i((int)value.X, (int)value.Y, (int)value.Z, (int)value.W);
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Vector4i other && Equals(other);
    }

    public readonly bool Equals(Vector4i other)
    {
        return X == other.X && Y == other.Y && Z == other.Z && W == other.W;
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(X, Y, Z, W);
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)}, {W.ToString(format, CultureInfo.InvariantCulture)})";
    }
}
}