using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Vector2 : IEquatable<Vector2>
{
    public enum Axis
    {
        X = 0,
        Y
    }

    public float X;
    public float Y;

    public float this[int index]
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

    public readonly void Deconstruct(out float x, out float y)
    {
        x = X;
        y = Y;
    }

    internal void Normalize()
    {
        float lengthsq = LengthSquared();

        if (lengthsq == 0)
        {
            X = Y = 0f;
        }
        else
        {
            float length = Mathf.Sqrt(lengthsq);
            X /= length;
            Y /= length;
        }
    }

    public readonly Vector2 Abs()
    {
        return new Vector2(Mathf.Abs(X), Mathf.Abs(Y));
    }

    public readonly float Angle()
    {
        return Mathf.Atan2(Y, X);
    }

    public readonly float AngleTo(Vector2 to)
    {
        return Mathf.Atan2(Cross(to), Dot(to));
    }

    public readonly float AngleToPoint(Vector2 to)
    {
        return Mathf.Atan2(to.Y - Y, to.X - X);
    }

    public readonly float Aspect()
    {
        return X / Y;
    }

    public readonly Vector2 Bounce(Vector2 normal)
    {
        return -Reflect(normal);
    }

    public readonly Vector2 Ceil()
    {
        return new Vector2(Mathf.Ceil(X), Mathf.Ceil(Y));
    }

    public readonly Vector2 Clamp(Vector2 min, Vector2 max)
    {
        return new Vector2
        (
            Mathf.Clamp(X, min.X, max.X),
            Mathf.Clamp(Y, min.Y, max.Y)
        );
    }

    public readonly Vector2 Clamp(float min, float max)
    {
        return new Vector2
        (
            Mathf.Clamp(X, min, max),
            Mathf.Clamp(Y, min, max)
        );
    }

    public readonly float Cross(Vector2 with)
    {
        return (X * with.Y) - (Y * with.X);
    }

    public readonly Vector2 CubicInterpolate(Vector2 b, Vector2 preA, Vector2 postB, float weight)
    {
        return new Vector2
        (
            Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight),
            Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight)
        );
    }

    public readonly Vector2 CubicInterpolateInTime(Vector2 b, Vector2 preA, Vector2 postB, float weight, float t, float preAT, float postBT)
    {
        return new Vector2
        (
            Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT),
            Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT)
        );
    }

    public readonly Vector2 BezierInterpolate(Vector2 control1, Vector2 control2, Vector2 end, float t)
    {
        return new Vector2
        (
            Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t),
            Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t)
        );
    }

    public readonly Vector2 BezierDerivative(Vector2 control1, Vector2 control2, Vector2 end, float t)
    {
        return new Vector2
        (
            Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t),
            Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t)
        );
    }

    public readonly Vector2 DirectionTo(Vector2 to)
    {
        return new Vector2(to.X - X, to.Y - Y).Normalized();
    }

    public readonly float DistanceSquaredTo(Vector2 to)
    {
        return (X - to.X) * (X - to.X) + (Y - to.Y) * (Y - to.Y);
    }

    public readonly float DistanceTo(Vector2 to)
    {
        return Mathf.Sqrt((X - to.X) * (X - to.X) + (Y - to.Y) * (Y - to.Y));
    }

    public readonly float Dot(Vector2 with)
    {
        return (X * with.X) + (Y * with.Y);
    }

    public readonly Vector2 Floor()
    {
        return new Vector2(Mathf.Floor(X), Mathf.Floor(Y));
    }

    public readonly Vector2 Inverse()
    {
        return new Vector2(1 / X, 1 / Y);
    }

    public readonly bool IsFinite()
    {
        return Mathf.IsFinite(X) && Mathf.IsFinite(Y);
    }

    public readonly bool IsNormalized()
    {
        return Mathf.IsEqualApprox(LengthSquared(), 1, Mathf.Epsilon);
    }

    public readonly float Length()
    {
        return Mathf.Sqrt((X * X) + (Y * Y));
    }

    public readonly float LengthSquared()
    {
        return (X * X) + (Y * Y);
    }

    public readonly Vector2 Lerp(Vector2 to, float weight)
    {
        return new Vector2
        (
            Mathf.Lerp(X, to.X, weight),
            Mathf.Lerp(Y, to.Y, weight)
        );
    }

    public readonly Vector2 LimitLength(float length = 1.0f)
    {
        Vector2 v = this;
        float l = Length();

        if (l > 0 && length < l)
        {
            v /= l;
            v *= length;
        }

        return v;
    }

    public readonly Vector2 Max(Vector2 with)
    {
        return new Vector2
        (
            Mathf.Max(X, with.X),
            Mathf.Max(Y, with.Y)
        );
    }

    public readonly Vector2 Max(float with)
    {
        return new Vector2
        (
            Mathf.Max(X, with),
            Mathf.Max(Y, with)
        );
    }

    public readonly Vector2 Min(Vector2 with)
    {
        return new Vector2
        (
            Mathf.Min(X, with.X),
            Mathf.Min(Y, with.Y)
        );
    }

    public readonly Vector2 Min(float with)
    {
        return new Vector2
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

    public readonly Vector2 MoveToward(Vector2 to, float delta)
    {
        Vector2 v = this;
        Vector2 vd = to - v;
        float len = vd.Length();
        if (len <= delta || len < Mathf.Epsilon)
            return to;

        return v + (vd / len * delta);
    }

    public readonly Vector2 Normalized()
    {
        Vector2 v = this;
        v.Normalize();
        return v;
    }

    public readonly Vector2 PosMod(float mod)
    {
        Vector2 v;
        v.X = Mathf.PosMod(X, mod);
        v.Y = Mathf.PosMod(Y, mod);
        return v;
    }

    public readonly Vector2 PosMod(Vector2 modv)
    {
        Vector2 v;
        v.X = Mathf.PosMod(X, modv.X);
        v.Y = Mathf.PosMod(Y, modv.Y);
        return v;
    }

    public readonly Vector2 Project(Vector2 onNormal)
    {
        return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
    }

    public readonly Vector2 Reflect(Vector2 normal)
    {
#if DEBUG
        if (!normal.IsNormalized())
        {
            throw new ArgumentException("Argument is not normalized.", nameof(normal));
        }
#endif
        return (2 * Dot(normal) * normal) - this;
    }

    public readonly Vector2 Rotated(float angle)
    {
        (float sin, float cos) = Mathf.SinCos(angle);
        return new Vector2
        (
            X * cos - Y * sin,
            X * sin + Y * cos
        );
    }

    public readonly Vector2 Round()
    {
        return new Vector2(Mathf.Round(X), Mathf.Round(Y));
    }

    public readonly Vector2 Sign()
    {
        Vector2 v;
        v.X = Mathf.Sign(X);
        v.Y = Mathf.Sign(Y);
        return v;
    }

    public readonly Vector2 Slerp(Vector2 to, float weight)
    {
        float startLengthSquared = LengthSquared();
        float endLengthSquared = to.LengthSquared();
        if (startLengthSquared == 0.0 || endLengthSquared == 0.0)
        {
            return Lerp(to, weight);
        }
        float startLength = Mathf.Sqrt(startLengthSquared);
        float resultLength = Mathf.Lerp(startLength, Mathf.Sqrt(endLengthSquared), weight);
        float angle = AngleTo(to);
        return Rotated(angle * weight) * (resultLength / startLength);
    }

    public readonly Vector2 Slide(Vector2 normal)
    {
        return this - (normal * Dot(normal));
    }

    public readonly Vector2 Snapped(Vector2 step)
    {
        return new Vector2(Mathf.Snapped(X, step.X), Mathf.Snapped(Y, step.Y));
    }

    public readonly Vector2 Snapped(float step)
    {
        return new Vector2(Mathf.Snapped(X, step), Mathf.Snapped(Y, step));
    }

    public readonly Vector2 Orthogonal()
    {
        return new Vector2(Y, -X);
    }

    private static readonly Vector2 _zero = new Vector2(0, 0);
    private static readonly Vector2 _one = new Vector2(1, 1);
    private static readonly Vector2 _inf = new Vector2(Mathf.Inf, Mathf.Inf);

    private static readonly Vector2 _up = new Vector2(0, -1);
    private static readonly Vector2 _down = new Vector2(0, 1);
    private static readonly Vector2 _right = new Vector2(1, 0);
    private static readonly Vector2 _left = new Vector2(-1, 0);

    public static Vector2 Zero { get { return _zero; } }
    public static Vector2 One { get { return _one; } }
    public static Vector2 Inf { get { return _inf; } }

    public static Vector2 Up { get { return _up; } }
    public static Vector2 Down { get { return _down; } }
    public static Vector2 Right { get { return _right; } }
    public static Vector2 Left { get { return _left; } }

    public Vector2(float x, float y)
    {
        X = x;
        Y = y;
    }

    public static Vector2 FromAngle(float angle)
    {
        (float sin, float cos) = Mathf.SinCos(angle);
        return new Vector2(cos, sin);
    }

    public static Vector2 operator +(Vector2 left, Vector2 right)
    {
        left.X += right.X;
        left.Y += right.Y;
        return left;
    }

    public static Vector2 operator -(Vector2 left, Vector2 right)
    {
        left.X -= right.X;
        left.Y -= right.Y;
        return left;
    }

    public static Vector2 operator -(Vector2 vec)
    {
        vec.X = -vec.X;
        vec.Y = -vec.Y;
        return vec;
    }

    public static Vector2 operator *(Vector2 vec, float scale)
    {
        vec.X *= scale;
        vec.Y *= scale;
        return vec;
    }

    public static Vector2 operator *(float scale, Vector2 vec)
    {
        vec.X *= scale;
        vec.Y *= scale;
        return vec;
    }

    public static Vector2 operator *(Vector2 left, Vector2 right)
    {
        left.X *= right.X;
        left.Y *= right.Y;
        return left;
    }

    public static Vector2 operator /(Vector2 vec, float divisor)
    {
        vec.X /= divisor;
        vec.Y /= divisor;
        return vec;
    }

    public static Vector2 operator /(Vector2 vec, Vector2 divisorv)
    {
        vec.X /= divisorv.X;
        vec.Y /= divisorv.Y;
        return vec;
    }

    public static Vector2 operator %(Vector2 vec, float divisor)
    {
        vec.X %= divisor;
        vec.Y %= divisor;
        return vec;
    }

    public static Vector2 operator %(Vector2 vec, Vector2 divisorv)
    {
        vec.X %= divisorv.X;
        vec.Y %= divisorv.Y;
        return vec;
    }

    public static bool operator ==(Vector2 left, Vector2 right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(Vector2 left, Vector2 right)
    {
        return !left.Equals(right);
    }

    public static bool operator <(Vector2 left, Vector2 right)
    {
        if (left.X == right.X)
        {
            return left.Y < right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >(Vector2 left, Vector2 right)
    {
        if (left.X == right.X)
        {
            return left.Y > right.Y;
        }
        return left.X > right.X;
    }

    public static bool operator <=(Vector2 left, Vector2 right)
    {
        if (left.X == right.X)
        {
            return left.Y <= right.Y;
        }
        return left.X < right.X;
    }

    public static bool operator >=(Vector2 left, Vector2 right)
    {
        if (left.X == right.X)
        {
            return left.Y >= right.Y;
        }
        return left.X > right.X;
    }

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Vector2 other && Equals(other);
    }

    public readonly bool Equals(Vector2 other)
    {
        return X == other.X && Y == other.Y;
    }

    public readonly bool IsEqualApprox(Vector2 other)
    {
        return Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y);
    }

    public readonly bool IsZeroApprox()
    {
        return Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y);
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