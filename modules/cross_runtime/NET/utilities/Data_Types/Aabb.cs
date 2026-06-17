using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
[Serializable]
[StructLayout(LayoutKind.Sequential)]
public struct AABB : IEquatable<AABB>
{
    public enum Axis
    {
        X = 0,
        Y = 1,
        Z = 2
    }

    private Vector3 _position;
    private Vector3 _size;

    public Vector3 Position
    {
        readonly get { return _position; }
        set { _position = value; }
    }

    public Vector3 Size
    {
        readonly get { return _size; }
        set { _size = value; }
    }

    public Vector3 End
    {
        readonly get { return Add(_position, _size); }
        set { _size = Sub(value, _position); }
    }

    public readonly float Volume
    {
        get { return _size.X * _size.Y * _size.Z; }
    }

    public readonly AABB Abs()
    {
        Vector3 end = End;
        Vector3 topLeft = Min(_position, end);
        return new AABB(topLeft, AbsVector(_size));
    }

    public readonly Vector3 GetCenter()
    {
        return Add(_position, Mul(_size, 0.5f));
    }

    public readonly bool Encloses(AABB with)
    {
        Vector3 srcMin = _position;
        Vector3 srcMax = Add(_position, _size);
        Vector3 dstMin = with._position;
        Vector3 dstMax = Add(with._position, with._size);

        return srcMin.X <= dstMin.X &&
               srcMax.X >= dstMax.X &&
               srcMin.Y <= dstMin.Y &&
               srcMax.Y >= dstMax.Y &&
               srcMin.Z <= dstMin.Z &&
               srcMax.Z >= dstMax.Z;
    }

    public readonly AABB Expand(Vector3 point)
    {
        Vector3 begin = _position;
        Vector3 end = Add(_position, _size);

        if (point.X < begin.X) begin.X = point.X;
        if (point.Y < begin.Y) begin.Y = point.Y;
        if (point.Z < begin.Z) begin.Z = point.Z;

        if (point.X > end.X) end.X = point.X;
        if (point.Y > end.Y) end.Y = point.Y;
        if (point.Z > end.Z) end.Z = point.Z;

        return new AABB(begin, Sub(end, begin));
    }

    public readonly Vector3 GetEndpoint(int idx)
    {
        switch (idx)
        {
            case 0: return new Vector3(_position.X, _position.Y, _position.Z);
            case 1: return new Vector3(_position.X, _position.Y, _position.Z + _size.Z);
            case 2: return new Vector3(_position.X, _position.Y + _size.Y, _position.Z);
            case 3: return new Vector3(_position.X, _position.Y + _size.Y, _position.Z + _size.Z);
            case 4: return new Vector3(_position.X + _size.X, _position.Y, _position.Z);
            case 5: return new Vector3(_position.X + _size.X, _position.Y, _position.Z + _size.Z);
            case 6: return new Vector3(_position.X + _size.X, _position.Y + _size.Y, _position.Z);
            case 7: return new Vector3(_position.X + _size.X, _position.Y + _size.Y, _position.Z + _size.Z);
            default:
                throw new ArgumentOutOfRangeException(nameof(idx),
                    $"Index is {idx}, but a value from 0 to 7 is expected.");
        }
    }

    public readonly Vector3 GetLongestAxis()
    {
        Vector3 axis = new Vector3(1f, 0f, 0f);
        float maxSize = _size.X;

        if (_size.Y > maxSize)
        {
            axis = new Vector3(0f, 1f, 0f);
            maxSize = _size.Y;
        }

        if (_size.Z > maxSize)
        {
            axis = new Vector3(0f, 0f, 1f);
        }

        return axis;
    }

    public readonly Axis GetLongestAxisIndex()
    {
        Axis axis = Axis.X;
        float maxSize = _size.X;

        if (_size.Y > maxSize)
        {
            axis = Axis.Y;
            maxSize = _size.Y;
        }

        if (_size.Z > maxSize)
        {
            axis = Axis.Z;
        }

        return axis;
    }

    public readonly float GetLongestAxisSize()
    {
        float maxSize = _size.X;

        if (_size.Y > maxSize)
            maxSize = _size.Y;

        if (_size.Z > maxSize)
            maxSize = _size.Z;

        return maxSize;
    }

    public readonly Vector3 GetShortestAxis()
    {
        Vector3 axis = new Vector3(1f, 0f, 0f);
        float minSize = _size.X;

        if (_size.Y < minSize)
        {
            axis = new Vector3(0f, 1f, 0f);
            minSize = _size.Y;
        }

        if (_size.Z < minSize)
        {
            axis = new Vector3(0f, 0f, 1f);
        }

        return axis;
    }

    public readonly Axis GetShortestAxisIndex()
    {
        Axis axis = Axis.X;
        float minSize = _size.X;

        if (_size.Y < minSize)
        {
            axis = Axis.Y;
            minSize = _size.Y;
        }

        if (_size.Z < minSize)
        {
            axis = Axis.Z;
        }

        return axis;
    }

    public readonly float GetShortestAxisSize()
    {
        float minSize = _size.X;

        if (_size.Y < minSize)
            minSize = _size.Y;

        if (_size.Z < minSize)
            minSize = _size.Z;

        return minSize;
    }

    public readonly Vector3 GetSupport(Vector3 dir)
    {
        Vector3 support = _position;

        if (dir.X > 0.0f) support.X += _size.X;
        if (dir.Y > 0.0f) support.Y += _size.Y;
        if (dir.Z > 0.0f) support.Z += _size.Z;

        return support;
    }

    public readonly AABB Grow(float by)
    {
        AABB res = this;

        res._position.X -= by;
        res._position.Y -= by;
        res._position.Z -= by;
        res._size.X += 2.0f * by;
        res._size.Y += 2.0f * by;
        res._size.Z += 2.0f * by;

        return res;
    }

    public readonly bool HasPoint(Vector3 point)
    {
        if (point.X < _position.X) return false;
        if (point.Y < _position.Y) return false;
        if (point.Z < _position.Z) return false;
        if (point.X > _position.X + _size.X) return false;
        if (point.Y > _position.Y + _size.Y) return false;
        if (point.Z > _position.Z + _size.Z) return false;

        return true;
    }

    public readonly bool HasSurface()
    {
        return _size.X > 0.0f || _size.Y > 0.0f || _size.Z > 0.0f;
    }

    public readonly bool HasVolume()
    {
        return _size.X > 0.0f && _size.Y > 0.0f && _size.Z > 0.0f;
    }

    public readonly AABB Intersection(AABB with)
    {
        Vector3 srcMin = _position;
        Vector3 srcMax = Add(_position, _size);
        Vector3 dstMin = with._position;
        Vector3 dstMax = Add(with._position, with._size);

        Vector3 min;
        Vector3 max;

        if (srcMin.X > dstMax.X || srcMax.X < dstMin.X)
            return new AABB();

        min.X = srcMin.X > dstMin.X ? srcMin.X : dstMin.X;
        max.X = srcMax.X < dstMax.X ? srcMax.X : dstMax.X;

        if (srcMin.Y > dstMax.Y || srcMax.Y < dstMin.Y)
            return new AABB();

        min.Y = srcMin.Y > dstMin.Y ? srcMin.Y : dstMin.Y;
        max.Y = srcMax.Y < dstMax.Y ? srcMax.Y : dstMax.Y;

        if (srcMin.Z > dstMax.Z || srcMax.Z < dstMin.Z)
            return new AABB();

        min.Z = srcMin.Z > dstMin.Z ? srcMin.Z : dstMin.Z;
        max.Z = srcMax.Z < dstMax.Z ? srcMax.Z : dstMax.Z;

        return new AABB(min, Sub(max, min));
    }

    public readonly bool Intersects(AABB with)
    {
        if (_position.X >= with._position.X + with._size.X) return false;
        if (_position.X + _size.X <= with._position.X) return false;
        if (_position.Y >= with._position.Y + with._size.Y) return false;
        if (_position.Y + _size.Y <= with._position.Y) return false;
        if (_position.Z >= with._position.Z + with._size.Z) return false;
        if (_position.Z + _size.Z <= with._position.Z) return false;

        return true;
    }

    public readonly bool IntersectsPlane(Plane plane)
    {
        Vector3 p0 = new Vector3(_position.X, _position.Y, _position.Z);
        Vector3 p1 = new Vector3(_position.X, _position.Y, _position.Z + _size.Z);
        Vector3 p2 = new Vector3(_position.X, _position.Y + _size.Y, _position.Z);
        Vector3 p3 = new Vector3(_position.X, _position.Y + _size.Y, _position.Z + _size.Z);
        Vector3 p4 = new Vector3(_position.X + _size.X, _position.Y, _position.Z);
        Vector3 p5 = new Vector3(_position.X + _size.X, _position.Y, _position.Z + _size.Z);
        Vector3 p6 = new Vector3(_position.X + _size.X, _position.Y + _size.Y, _position.Z);
        Vector3 p7 = new Vector3(_position.X + _size.X, _position.Y + _size.Y, _position.Z + _size.Z);

        bool over = false;
        bool under = false;

        Vector3[] points = [p0, p1, p2, p3, p4, p5, p6, p7];

        for (int i = 0; i < 8; i++)
        {
            float distance = PlaneDistanceTo(plane, points[i]);
            if (distance > 0.0f)
                over = true;
            else
                under = true;
        }

        return under && over;
    }

    public readonly bool IntersectsSegment(Vector3 from, Vector3 to)
    {
        float min = 0.0f;
        float max = 1.0f;

        for (int i = 0; i < 3; i++)
        {
            float segFrom = GetAxis(from, i);
            float segTo = GetAxis(to, i);
            float boxBegin = GetAxis(_position, i);
            float boxEnd = boxBegin + GetAxis(_size, i);
            float cmin;
            float cmax;

            if (segFrom < segTo)
            {
                if (segFrom > boxEnd || segTo < boxBegin)
                    return false;

                float length = segTo - segFrom;
                cmin = segFrom < boxBegin ? (boxBegin - segFrom) / length : 0.0f;
                cmax = segTo > boxEnd ? (boxEnd - segFrom) / length : 1.0f;
            }
            else
            {
                if (segTo > boxEnd || segFrom < boxBegin)
                    return false;

                float length = segTo - segFrom;
                cmin = segFrom > boxEnd ? (boxEnd - segFrom) / length : 0.0f;
                cmax = segTo < boxBegin ? (boxBegin - segFrom) / length : 1.0f;
            }

            if (cmin > min)
                min = cmin;

            if (cmax < max)
                max = cmax;

            if (max < min)
                return false;
        }

        return true;
    }

    public readonly bool IsFinite()
    {
        return IsFinite(_position.X) && IsFinite(_position.Y) && IsFinite(_position.Z) &&
               IsFinite(_size.X) && IsFinite(_size.Y) && IsFinite(_size.Z);
    }

    public readonly AABB Merge(AABB with)
    {
        Vector3 beg1 = _position;
        Vector3 beg2 = with._position;
        Vector3 end1 = Add(beg1, _size);
        Vector3 end2 = Add(beg2, with._size);

        Vector3 min = new Vector3(
            beg1.X < beg2.X ? beg1.X : beg2.X,
            beg1.Y < beg2.Y ? beg1.Y : beg2.Y,
            beg1.Z < beg2.Z ? beg1.Z : beg2.Z
        );

        Vector3 max = new Vector3(
            end1.X > end2.X ? end1.X : end2.X,
            end1.Y > end2.Y ? end1.Y : end2.Y,
            end1.Z > end2.Z ? end1.Z : end2.Z
        );

        return new AABB(min, Sub(max, min));
    }

    public AABB(Vector3 position, Vector3 size)
    {
        _position = position;
        _size = size;
    }

    public AABB(float x, float y, float z, Vector3 size)
    {
        _position = new Vector3(x, y, z);
        _size = size;
    }

    public AABB(float x, float y, float z, float width, float height, float depth)
    {
        _position = new Vector3(x, y, z);
        _size = new Vector3(width, height, depth);
    }

    public static bool operator ==(AABB left, AABB right) => left.Equals(right);
    public static bool operator !=(AABB left, AABB right) => !left.Equals(right);

    public override readonly bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is AABB other && Equals(other);
    }

    public readonly bool Equals(AABB other)
    {
        return _position.X == other._position.X &&
               _position.Y == other._position.Y &&
               _position.Z == other._position.Z &&
               _size.X == other._size.X &&
               _size.Y == other._size.Y &&
               _size.Z == other._size.Z;
    }

    public override readonly int GetHashCode()
    {
        return HashCode.Combine(
            _position.X, _position.Y, _position.Z,
            _size.X, _size.Y, _size.Z
        );
    }

    public override readonly string ToString() => ToString(null);

    public readonly string ToString(string? format)
    {
        string fmt = string.IsNullOrEmpty(format) ? "G" : format;

        return string.Create(CultureInfo.InvariantCulture, $"{FormatVec3(_position, fmt)}, {FormatVec3(_size, fmt)}");
    }

    private static Vector3 Add(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
    }

    private static Vector3 Sub(Vector3 a, Vector3 b)
    {
        return new Vector3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
    }

    private static Vector3 Mul(Vector3 a, float b)
    {
        return new Vector3(a.X * b, a.Y * b, a.Z * b);
    }

    private static Vector3 Min(Vector3 a, Vector3 b)
    {
        return new Vector3(
            a.X < b.X ? a.X : b.X,
            a.Y < b.Y ? a.Y : b.Y,
            a.Z < b.Z ? a.Z : b.Z
        );
    }

    private static Vector3 AbsVector(Vector3 v)
    {
        return new Vector3(MathF.Abs(v.X), MathF.Abs(v.Y), MathF.Abs(v.Z));
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static float GetAxis(Vector3 v, int axis)
    {
        return axis switch
        {
            0 => v.X,
            1 => v.Y,
            2 => v.Z,
            _ => throw new ArgumentOutOfRangeException(nameof(axis))
        };
    }

    private static float PlaneDistanceTo(Plane plane, Vector3 point)
    {
        return plane.Normal.X * point.X +
               plane.Normal.Y * point.Y +
               plane.Normal.Z * point.Z +
               plane.D;
    }

    private static string FormatVec3(Vector3 v, string format)
    {
        return "(" +
               v.X.ToString(format, CultureInfo.InvariantCulture) + ", " +
               v.Y.ToString(format, CultureInfo.InvariantCulture) + ", " +
               v.Z.ToString(format, CultureInfo.InvariantCulture) + ")";
    }
}
}