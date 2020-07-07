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
        /// <summary>
        /// Enumerated index values for the axes.
        /// Returned by <see cref="MaxAxis"/> and <see cref="MinAxis"/>.
        /// </summary>
        public enum Axis
        {
            X = 0,
            Y,
            Z
        }

        /// <summary>
        /// The vector's X component. Also accessible by using the index position `[0]`.
        /// </summary>
        public int x;
        /// <summary>
        /// The vector's Y component. Also accessible by using the index position `[1]`.
        /// </summary>
        public int y;
        /// <summary>
        /// The vector's Z component. Also accessible by using the index position `[2]`.
        /// </summary>
        public int z;

        /// <summary>
        /// Access vector components using their index.
        /// </summary>
        /// <value>`[0]` is equivalent to `.x`, `[1]` is equivalent to `.y`, `[2]` is equivalent to `.z`.</value>
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

        /// <summary>
        /// Returns a new vector with all components in absolute values (i.e. positive).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Abs(int)"/> called on each component.</returns>
        public Vector3i Abs()
        {
            return new Vector3i(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z));
        }

        /// <summary>
        /// Returns the squared distance between this vector and `b`.
        /// This method runs faster than <see cref="DistanceTo"/>, so prefer it if
        /// you need to compare vectors or need the squared distance for some formula.
        /// </summary>
        /// <param name="b">The other vector to use.</param>
        /// <returns>The squared distance between the two vectors.</returns>
        public int DistanceSquaredTo(Vector3i b)
        {
            return (b - this).LengthSquared();
        }

        /// <summary>
        /// Returns the distance between this vector and `b`.
        /// </summary>
        /// <param name="b">The other vector to use.</param>
        /// <returns>The distance between the two vectors.</returns>
        public real_t DistanceTo(Vector3i b)
        {
            return (b - this).Length();
        }

        /// <summary>
        /// Returns the dot product of this vector and `b`.
        /// </summary>
        /// <param name="b">The other vector to use.</param>
        /// <returns>The dot product of the two vectors.</returns>
        public int Dot(Vector3i b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        /// <summary>
        /// Returns the length (magnitude) of this vector.
        /// </summary>
        /// <returns>The length of this vector.</returns>
        public real_t Length()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        /// <summary>
        /// Returns the squared length (squared magnitude) of this vector.
        /// This method runs faster than <see cref="Length"/>, so prefer it if
        /// you need to compare vectors or need the squared length for some formula.
        /// </summary>
        /// <returns>The squared length of this vector.</returns>
        public int LengthSquared()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return x2 + y2 + z2;
        }

        /// <summary>
        /// Returns the axis of the vector's largest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.X"/>.
        /// </summary>
        /// <returns>The index of the largest axis.</returns>
        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        /// <summary>
        /// Returns the axis of the vector's smallest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.Z"/>.
        /// </summary>
        /// <returns>The index of the smallest axis.</returns>
        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        /// <summary>
        /// Returns a vector composed of the <see cref="Mathf.PosMod(int, int)"/> of this vector's components and `mod`.
        /// </summary>
        /// <param name="mod">A value representing the divisor of the operation.</param>
        /// <returns>A vector with each component <see cref="Mathf.PosMod(int, int)"/> by `mod`.</returns>
        public Vector3i PosMod(int mod)
        {
            Vector3i v = this;
            v.x = Mathf.PosMod(v.x, mod);
            v.y = Mathf.PosMod(v.y, mod);
            v.z = Mathf.PosMod(v.z, mod);
            return v;
        }

        /// <summary>
        /// Returns a vector composed of the <see cref="Mathf.PosMod(int, int)"/> of this vector's components and `modv`'s components.
        /// </summary>
        /// <param name="modv">A vector representing the divisors of the operation.</param>
        /// <returns>A vector with each component <see cref="Mathf.PosMod(int, int)"/> by `modv`'s components.</returns>
        public Vector3i PosMod(Vector3i modv)
        {
            Vector3i v = this;
            v.x = Mathf.PosMod(v.x, modv.x);
            v.y = Mathf.PosMod(v.y, modv.y);
            v.z = Mathf.PosMod(v.z, modv.z);
            return v;
        }

        /// <summary>
        /// Returns a vector with each component set to one or negative one, depending
        /// on the signs of this vector's components, or zero if the component is zero,
        /// by calling <see cref="Mathf.Sign(int)"/> on each component.
        /// </summary>
        /// <returns>A vector with all components as either `1`, `-1`, or `0`.</returns>
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

        /// <summary>
        /// Zero vector, a vector with all components set to `0`.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(0, 0, 0)`</value>
        public static Vector3i Zero { get { return _zero; } }
        /// <summary>
        /// One vector, a vector with all components set to `1`.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(1, 1, 1)`</value>
        public static Vector3i One { get { return _one; } }

        /// <summary>
        /// Up unit vector.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(0, 1, 0)`</value>
        public static Vector3i Up { get { return _up; } }
        /// <summary>
        /// Down unit vector.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(0, -1, 0)`</value>
        public static Vector3i Down { get { return _down; } }
        /// <summary>
        /// Right unit vector. Represents the local direction of right,
        /// and the global direction of east.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(1, 0, 0)`</value>
        public static Vector3i Right { get { return _right; } }
        /// <summary>
        /// Left unit vector. Represents the local direction of left,
        /// and the global direction of west.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(-1, 0, 0)`</value>
        public static Vector3i Left { get { return _left; } }
        /// <summary>
        /// Forward unit vector. Represents the local direction of forward,
        /// and the global direction of north.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(0, 0, -1)`</value>
        public static Vector3i Forward { get { return _forward; } }
        /// <summary>
        /// Back unit vector. Represents the local direction of back,
        /// and the global direction of south.
        /// </summary>
        /// <value>Equivalent to `new Vector3i(0, 0, 1)`</value>
        public static Vector3i Back { get { return _back; } }

        /// <summary>
        /// Constructs a new <see cref="Vector3i"/> with the given components.
        /// </summary>
        /// <param name="x">The vector's X component.</param>
        /// <param name="y">The vector's Y component.</param>
        /// <param name="z">The vector's Z component.</param>
        public Vector3i(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        /// <summary>
        /// Constructs a new <see cref="Vector3i"/> from an existing <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="vi">The existing <see cref="Vector3i"/>.</param>
        public Vector3i(Vector3i vi)
        {
            this.x = vi.x;
            this.y = vi.y;
            this.z = vi.z;
        }

        /// <summary>
        /// Constructs a new <see cref="Vector3i"/> from an existing <see cref="Vector3"/>
        /// by rounding the components via <see cref="Mathf.RoundToInt(real_t)"/>.
        /// </summary>
        /// <param name="v">The <see cref="Vector3"/> to convert.</param>
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
