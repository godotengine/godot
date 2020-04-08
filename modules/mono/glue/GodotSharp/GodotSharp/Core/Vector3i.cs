using System;
using System.Runtime.InteropServices;

#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    /// <summary>
    /// 3-element structure that can be used to represent 3D grid coordinates or sets of integers.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3i : IEquatable<Vector3i>
    {
        public enum Axis
        {
            X = 0,
            Y,
            Z
        }

        public int x;
        public int y;
        public int z;

        public int this[int index]
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

        public Vector3i Abs()
        {
            Vector3i v = this;
            if (v.x < 0)
            {
                v.x = -v.x;
            }
            if (v.y < 0)
            {
                v.y = -v.y;
            }
            if (v.z < 0)
            {
                v.z = -v.z;
            }
            return v;
        }

        public int DistanceSquaredTo(Vector3i b)
        {
            return (b - this).LengthSquared();
        }

        public real_t DistanceTo(Vector3i b)
        {
            return (b - this).Length();
        }

        public int Dot(Vector3i b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        public real_t Length()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        public int LengthSquared()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return x2 + y2 + z2;
        }

        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        public Vector3i PosMod(int mod)
        {
            Vector3i v = this;
            v.x = Mathf.PosMod(v.x, mod);
            v.y = Mathf.PosMod(v.y, mod);
            v.z = Mathf.PosMod(v.z, mod);
            return v;
        }

        public Vector3i PosMod(Vector3i modv)
        {
            Vector3i v = this;
            v.x = Mathf.PosMod(v.x, modv.x);
            v.y = Mathf.PosMod(v.y, modv.y);
            v.z = Mathf.PosMod(v.z, modv.z);
            return v;
        }

        public Vector3i Sign()
        {
            Vector3i v = this;
            v.x = Mathf.Sign(v.x);
            v.y = Mathf.Sign(v.y);
            v.z = Mathf.Sign(v.z);
            return v;
        }

        // Constants
        private static readonly Vector3i _zero = new Vector3i(0, 0, 0);
        private static readonly Vector3i _one = new Vector3i(1, 1, 1);

        private static readonly Vector3i _up = new Vector3i(0, 1, 0);
        private static readonly Vector3i _down = new Vector3i(0, -1, 0);
        private static readonly Vector3i _right = new Vector3i(1, 0, 0);
        private static readonly Vector3i _left = new Vector3i(-1, 0, 0);
        private static readonly Vector3i _forward = new Vector3i(0, 0, -1);
        private static readonly Vector3i _back = new Vector3i(0, 0, 1);

        public static Vector3i Zero { get { return _zero; } }
        public static Vector3i One { get { return _one; } }

        public static Vector3i Up { get { return _up; } }
        public static Vector3i Down { get { return _down; } }
        public static Vector3i Right { get { return _right; } }
        public static Vector3i Left { get { return _left; } }
        public static Vector3i Forward { get { return _forward; } }
        public static Vector3i Back { get { return _back; } }

        // Constructors
        public Vector3i(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public Vector3i(Vector3i vi)
        {
            this.x = vi.x;
            this.y = vi.y;
            this.z = vi.z;
        }
        public Vector3i(Vector3 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
            this.z = Mathf.RoundToInt(v.z);
        }

        public static Vector3i operator +(Vector3i left, Vector3i right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        public static Vector3i operator -(Vector3i left, Vector3i right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        public static Vector3i operator -(Vector3i vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            return vec;
        }

        public static Vector3i operator *(Vector3i vec, int scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3i operator *(int scale, Vector3i vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3i operator *(Vector3i left, Vector3i right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        public static Vector3i operator /(Vector3i vec, int divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            return vec;
        }

        public static Vector3i operator /(Vector3i vec, Vector3i divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            return vec;
        }

        public static Vector3i operator %(Vector3i vec, int divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            vec.z %= divisor;
            return vec;
        }

        public static Vector3i operator %(Vector3i vec, Vector3i divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            vec.z %= divisorv.z;
            return vec;
        }

        public static Vector3i operator &(Vector3i vec, int and)
        {
            vec.x &= and;
            vec.y &= and;
            vec.z &= and;
            return vec;
        }

        public static Vector3i operator &(Vector3i vec, Vector3i andv)
        {
            vec.x &= andv.x;
            vec.y &= andv.y;
            vec.z &= andv.z;
            return vec;
        }

        public static bool operator ==(Vector3i left, Vector3i right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector3i left, Vector3i right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Vector3i left, Vector3i right)
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

        public static bool operator >(Vector3i left, Vector3i right)
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

        public static bool operator <=(Vector3i left, Vector3i right)
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

        public static bool operator >=(Vector3i left, Vector3i right)
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

        public static implicit operator Vector3(Vector3i value)
        {
            return new Vector3(value.x, value.y, value.z);
        }

        public static explicit operator Vector3i(Vector3 value)
        {
            return new Vector3i(value);
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3i)
            {
                return Equals((Vector3i)obj);
            }

            return false;
        }

        public bool Equals(Vector3i other)
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
