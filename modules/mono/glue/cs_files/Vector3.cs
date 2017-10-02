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

        internal void normalize()
        {
            float length = this.length();

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

        public Vector3 abs()
        {
            return new Vector3(Mathf.abs(x), Mathf.abs(y), Mathf.abs(z));
        }

        public float angle_to(Vector3 to)
        {
            return Mathf.atan2(cross(to).length(), dot(to));
        }

        public Vector3 bounce(Vector3 n)
        {
            return -reflect(n);
        }

        public Vector3 ceil()
        {
            return new Vector3(Mathf.ceil(x), Mathf.ceil(y), Mathf.ceil(z));
        }

        public Vector3 cross(Vector3 b)
        {
            return new Vector3
            (
                (y * b.z) - (z * b.y),
                (z * b.x) - (x * b.z),
                (x * b.y) - (y * b.x)
            );
        }

        public Vector3 cubic_interpolate(Vector3 b, Vector3 preA, Vector3 postB, float t)
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

        public float distance_squared_to(Vector3 b)
        {
            return (b - this).length_squared();
        }

        public float distance_to(Vector3 b)
        {
            return (b - this).length();
        }

        public float dot(Vector3 b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        public Vector3 floor()
        {
            return new Vector3(Mathf.floor(x), Mathf.floor(y), Mathf.floor(z));
        }

        public Vector3 inverse()
        {
            return new Vector3(1.0f / x, 1.0f / y, 1.0f / z);
        }

        public bool is_normalized()
        {
            return Mathf.abs(length_squared() - 1.0f) < Mathf.Epsilon;
        }

        public float length()
        {
            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;

            return Mathf.sqrt(x2 + y2 + z2);
        }

        public float length_squared()
        {
            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;

            return x2 + y2 + z2;
        }

        public Vector3 linear_interpolate(Vector3 b, float t)
        {
            return new Vector3
            (
                x + (t * (b.x - x)),
                y + (t * (b.y - y)),
                z + (t * (b.z - z))
            );
        }

        public Axis max_axis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        public Axis min_axis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        public Vector3 normalized()
        {
            Vector3 v = this;
            v.normalize();
            return v;
        }

        public Basis outer(Vector3 b)
        {
            return new Basis(
                new Vector3(x * b.x, x * b.y, x * b.z),
                new Vector3(y * b.x, y * b.y, y * b.z),
                new Vector3(z * b.x, z * b.y, z * b.z)
            );
        }

        public Vector3 reflect(Vector3 n)
        {
#if DEBUG
            if (!n.is_normalized())
                throw new ArgumentException(String.Format("{0} is not normalized", n), nameof(n));
#endif
            return 2.0f * n * dot(n) - this;
        }

        public Vector3 rotated(Vector3 axis, float phi)
        {
            return new Basis(axis, phi).xform(this);
        }

        public Vector3 slide(Vector3 n)
        {
            return this - n * dot(n);
        }

        public Vector3 snapped(Vector3 by)
        {
            return new Vector3
            (
                Mathf.stepify(x, by.x),
                Mathf.stepify(y, by.y),
                Mathf.stepify(z, by.z)
            );
        }

        public Basis to_diagonal_matrix()
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
