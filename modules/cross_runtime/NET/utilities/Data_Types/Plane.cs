using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
/// <summary>
/// Plane represents a normalized plane equation.

/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct Plane : IEquatable<Plane>
{
    private Vector3 _normal;
    private float _d;

    public Vector3 Normal
    {
        readonly get { return _normal; }
        set { _normal = value; }
    }

    public float D
    {
        readonly get { return _d; }
        set { _d = value; }
    }

    public float X
    {
        readonly get { return _normal.X; }
        set { _normal.X = value; }
    }

    public float Y
    {
        readonly get { return _normal.Y; }
        set { _normal.Y = value; }
    }

    public float Z
    {
        readonly get { return _normal.Z; }
        set { _normal.Z = value; }
    }

    public readonly float DistanceTo(Vector3 point)
    {
        return Dot(_normal, point) - _d;
    }

    public readonly Vector3 GetCenter()
    {
        return Mul(_normal, _d);
    }

    public readonly bool HasPoint(Vector3 point, float tolerance = 0.00001f)
    {
        float dist = Dot(_normal, point) - _d;
        return MathF.Abs(dist) <= tolerance;
    }

    public readonly Vector3? Intersect3(Plane b, Plane c)
    {
        Vector3 n1xn2 = Cross(_normal, b._normal);
        float denom = Dot(n1xn2, c._normal);

        if (IsZeroApprox(denom))
            return null;

        Vector3 term1 = Mul(Cross(b._normal, c._normal), _d);
        Vector3 term2 = Mul(Cross(c._normal, _normal), b._d);
        Vector3 term3 = Mul(Cross(_normal, b._normal), c._d);

        Vector3 result = Add(Add(term1, term2), term3);
        return Div(result, denom);
    }

    public readonly Vector3? IntersectsRay(Vector3 from, Vector3 dir)
    {
        float den = Dot(_normal, dir);

        if (IsZeroApprox(den))
            return null;

        float dist = (Dot(_normal, from) - _d) / den;

        if (dist > 0.00001f)
            return null;

        return Sub(from, Mul(dir, dist));
    }

    public readonly Vector3? IntersectsSegment(Vector3 begin, Vector3 end)
    {
        Vector3 segment = Sub(begin, end);
        float den = Dot(_normal, segment);

        if (IsZeroApprox(den))
            return null;

        float dist = (Dot(_normal, begin) - _d) / den;

        if (dist < -0.00001f || dist > 1.0f + 0.00001f)
            return null;

        return Sub(begin, Mul(segment, dist));
    }

    public readonly bool IsFinite()
    {
        return IsFinite(_normal.X) && IsFinite(_normal.Y) && IsFinite(_normal.Z) && IsFinite(_d);
    }

    public readonly bool IsPointOver(Vector3 point)
    {
        return Dot(_normal, point) > _d;
    }

    public readonly Plane Normalized()
    {
        float len = Length(_normal);

        if (len == 0f)
            return new Plane(0f, 0f, 0f, 0f);

        return new Plane(Div(_normal, len), _d / len);
    }

    public readonly Vector3 Project(Vector3 point)
    {
        return Sub(point, Mul(_normal, DistanceTo(point)));
    }

    private static readonly Plane _planeYZ = new Plane(1f, 0f, 0f, 0f);
    private static readonly Plane _planeXZ = new Plane(0f, 1f, 0f, 0f);
    private static readonly Plane _planeXY = new Plane(0f, 0f, 1f, 0f);

    public static Plane PlaneYZ => _planeYZ;
    public static Plane PlaneXZ => _planeXZ;
    public static Plane PlaneXY => _planeXY;

    public Plane(float a, float b, float c, float d)
    {
        _normal = new Vector3(a, b, c);
        _d = d;
    }

    public Plane(Vector3 normal)
    {
        _normal = normal;
        _d = 0f;
    }

    public Plane(Vector3 normal, float d)
    {
        _normal = normal;
        _d = d;
    }

    public Plane(Vector3 normal, Vector3 point)
    {
        _normal = normal;
        _d = Dot(normal, point);
    }

    public Plane(Vector3 v1, Vector3 v2, Vector3 v3)
    {
        Vector3 a = Sub(v1, v3);
        Vector3 b = Sub(v1, v2);
        _normal = Normalize(Cross(a, b));
        _d = Dot(_normal, v1);
    }

    public static Plane operator -(Plane plane)
    {
        return new Plane(Neg(plane._normal), -plane._d);
    }

    public static bool operator ==(Plane left, Plane right) => left.Equals(right);
    public static bool operator !=(Plane left, Plane right) => !left.Equals(right);

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is Plane other && Equals(other);
    }

    public readonly bool Equals(Plane other)
    {
        return _normal.X == other._normal.X &&
               _normal.Y == other._normal.Y &&
               _normal.Z == other._normal.Z &&
               _d == other._d;
    }

    public readonly bool IsEqualApprox(Plane other)
    {
        return IsEqualApprox(_normal.X, other._normal.X) &&
               IsEqualApprox(_normal.Y, other._normal.Y) &&
               IsEqualApprox(_normal.Z, other._normal.Z) &&
               IsEqualApprox(_d, other._d);
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(_normal.X, _normal.Y, _normal.Z, _d);
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        string f = string.IsNullOrEmpty(format) ? "G" : format!;
        return $"{_normal.X.ToString(f, CultureInfo.InvariantCulture)}, {_normal.Y.ToString(f, CultureInfo.InvariantCulture)}, {_normal.Z.ToString(f, CultureInfo.InvariantCulture)}, {_d.ToString(f, CultureInfo.InvariantCulture)}";
    }

    private static Vector3 Add(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    private static Vector3 Sub(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    }

    private static Vector3 Mul(Vector3 v, float s)
    {
        return new Vector3(v.X * s, v.Y * s, v.Z * s);
    }

    private static Vector3 Div(Vector3 v, float s)
    {
        return new Vector3(v.X / s, v.Y / s, v.Z / s);
    }

    private static Vector3 Neg(Vector3 v)
    {
        return new Vector3(-v.X, -v.Y, -v.Z);
    }

    private static float Dot(Vector3 a, Vector3 b)
    {
        return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    }

    private static Vector3 Cross(Vector3 a, Vector3 b)
    {
        return new Vector3(
            a.Y * b.Z - a.Z * b.Y,
            a.Z * b.X - a.X * b.Z,
            a.X * b.Y - a.Y * b.X
        );
    }

    private static float Length(Vector3 v)
    {
        return MathF.Sqrt(Dot(v, v));
    }

    private static Vector3 Normalize(Vector3 v)
    {
        float len = Length(v);
        if (len == 0f)
            return new Vector3(0f, 0f, 0f);

        return Div(v, len);
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool IsZeroApprox(float value)
    {
        return MathF.Abs(value) <= 0.00001f;
    }

    private static bool IsEqualApprox(float a, float b)
    {
        return MathF.Abs(a - b) <= 0.00001f;
    }
}
}