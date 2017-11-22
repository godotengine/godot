using System;
using System.Runtime.InteropServices;

// file: core/math/vector3.h
// commit: bd282ff43f23fe845f29a3e25c8efc01bd65ffb0
// file: core/math/vector3.cpp
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/variant_call.cpp
// commit: 5ad9be4c24e9d7dc5672fdc42cea896622fe5685

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3 : IEquatable<Vector3>
    {
        public enum Axis
        {
            X = 0,
            Y,
            Z
        }

        public float x;
        public float y;
        public float z;

        public float this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return z;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        z = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        internal void Normalize()
        {
            float length = this.Length();

            if (length == 0f)
            {
                x = y = z = 0f;
            }
            else
            {
                x /= length;
                y /= length;
                z /= length;
            }
        }

        public Vector3 Abs()
        {
            return new Vector3(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z));
        }

        public float AngleTo(Vector3 to)
        {
            return Mathf.Atan2(Cross(to).Length(), Dot(to));
        }

        public Vector3 Bounce(Vector3 n)
        {
            return -Reflect(n);
        }

        public Vector3 Ceil()
        {
            return new Vector3(Mathf.Ceil(x), Mathf.Ceil(y), Mathf.Ceil(z));
        }

        public Vector3 Cross(Vector3 b)
        {
            return new Vector3
            (
                (y * b.z) - (z * b.y),
                (z * b.x) - (x * b.z),
                (x * b.y) - (y * b.x)
            );
        }

        public Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, float t)
        {
            Vector3 p0 = preA;
            Vector3 p1 = this;
            Vector3 p2 = b;
            Vector3 p3 = postB;

            float t2 = t * t;
            float t3 = t2 * t;

            return 0.5f * (
                        (p1 * 2.0f) + (-p0 + p2) * t +
                        (2.0f * p0 - 5.0f * p1 + 4f * p2 - p3) * t2 +
                        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
                    );
        }

        public float DistanceSquaredTo(Vector3 b)
        {
            return (b - this).LengthSquared();
        }

        public float DistanceTo(Vector3 b)
        {
            return (b - this).Length();
        }

        public float Dot(Vector3 b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        public Vector3 Floor()
        {
            return new Vector3(Mathf.Floor(x), Mathf.Floor(y), Mathf.Floor(z));
        }

        public Vector3 Inverse()
        {
            return new Vector3(1.0f / x, 1.0f / y, 1.0f / z);
        }

        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared() - 1.0f) < Mathf.Epsilon;
        }

        public float Length()
        {
            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        public float LengthSquared()
        {
            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;

            return x2 + y2 + z2;
        }

        public Vector3 LinearInterpolate(Vector3 b, float t)
        {
            return new Vector3
            (
                x + (t * (b.x - x)),
                y + (t * (b.y - y)),
                z + (t * (b.z - z))
            );
        }

        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        public Vector3 Normalized()
        {
            Vector3 v = this;
            v.Normalize();
            return v;
        }

        public Basis Outer(Vector3 b)
        {
            return new Basis(
                new Vector3(x * b.x, x * b.y, x * b.z),
                new Vector3(y * b.x, y * b.y, y * b.z),
                new Vector3(z * b.x, z * b.y, z * b.z)
            );
        }

        public Vector3 Reflect(Vector3 n)
        {
#if DEBUG
            if (!n.IsNormalized())
                throw new ArgumentException(String.Format("{0} is not normalized", n), nameof(n));
#endif
            return 2.0f * n * Dot(n) - this;
        }

        public Vector3 Rotated(Vector3 axis, float phi)
        {
            return new Basis(axis, phi).Xform(this);
        }

        public Vector3 Slide(Vector3 n)
        {
            return this - n * Dot(n);
        }

        public Vector3 Snapped(Vector3 by)
        {
            return new Vector3
            (
                Mathf.Stepify(x, by.x),
                Mathf.Stepify(y, by.y),
                Mathf.Stepify(z, by.z)
            );
        }

        public Basis ToDiagonalMatrix()
        {
            return new Basis(
                x, 0f, 0f,
                0f, y, 0f,
                0f, 0f, z
            );
        }

        public Vector3(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public static Vector3 operator +(Vector3 left, Vector3 right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        public static Vector3 operator -(Vector3 left, Vector3 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        public static Vector3 operator -(Vector3 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            return vec;
        }

        public static Vector3 operator *(Vector3 vec, float scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3 operator *(float scale, Vector3 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3 operator *(Vector3 left, Vector3 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        public static Vector3 operator /(Vector3 vec, float scale)
        {
            vec.x /= scale;
            vec.y /= scale;
            vec.z /= scale;
            return vec;
        }

        public static Vector3 operator /(Vector3 left, Vector3 right)
        {
            left.x /= right.x;
            left.y /= right.y;
            left.z /= right.z;
            return left;
        }

        public static bool operator ==(Vector3 left, Vector3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector3 left, Vector3 right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z < right.z;
                else
                    return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z > right.z;
                else
                    return left.y > right.y;
            }

            return left.x > right.x;
        }

        public static bool operator <=(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z <= right.z;
                else
                    return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >=(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z >= right.z;
                else
                    return left.y > right.y;
            }

            return left.x > right.x;
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3)
            {
                return Equals((Vector3)obj);
            }

            return false;
        }

        public bool Equals(Vector3 other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(),
                this.y.ToString(),
                this.z.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format),
                this.z.ToString(format)
            });
        }
    }
}
