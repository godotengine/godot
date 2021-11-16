#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Runtime.InteropServices;

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
            /// <summary>
            /// The vector's X axis.
            /// </summary>
            X = 0,
            /// <summary>
            /// The vector's Y axis.
            /// </summary>
            Y,
            /// <summary>
            /// The vector's Z axis.
            /// </summary>
            Z
        }

        /// <summary>
        /// The vector's X component. Also accessible by using the index position <c>[0]</c>.
        /// </summary>
        public int x;

        /// <summary>
        /// The vector's Y component. Also accessible by using the index position <c>[1]</c>.
        /// </summary>
        public int y;

        /// <summary>
        /// The vector's Z component. Also accessible by using the index position <c>[2]</c>.
        /// </summary>
        public int z;

        /// <summary>
        /// Access vector components using their <paramref name="index"/>.
        /// </summary>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the given the <paramref name="index"/> is not 0, 1 or 2.
        /// </exception>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="x"/>,
        /// <c>[1]</c> is equivalent to <see cref="y"/>,
        /// <c>[2]</c> is equivalent to <see cref="z"/>.
        /// </value>
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
        /// Returns a new vector with all components clamped between the
        /// components of <paramref name="min"/> and <paramref name="max"/> using
        /// <see cref="Mathf.Clamp(int, int, int)"/>.
        /// </summary>
        /// <param name="min">The vector with minimum allowed values.</param>
        /// <param name="max">The vector with maximum allowed values.</param>
        /// <returns>The vector with all components clamped.</returns>
        public Vector3i Clamp(Vector3i min, Vector3i max)
        {
            return new Vector3i
            (
                Mathf.Clamp(x, min.x, max.x),
                Mathf.Clamp(y, min.y, max.y),
                Mathf.Clamp(z, min.z, max.z)
            );
        }

        /// <summary>
        /// Returns the squared distance between this vector and <paramref name="b"/>.
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
        /// Returns the distance between this vector and <paramref name="b"/>.
        /// </summary>
        /// <seealso cref="DistanceSquaredTo(Vector3i)"/>
        /// <param name="b">The other vector to use.</param>
        /// <returns>The distance between the two vectors.</returns>
        public real_t DistanceTo(Vector3i b)
        {
            return (b - this).Length();
        }

        /// <summary>
        /// Returns the dot product of this vector and <paramref name="b"/>.
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
        /// <seealso cref="LengthSquared"/>
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
        /// Returns a vector composed of the <see cref="Mathf.PosMod(int, int)"/> of this vector's components
        /// and <paramref name="mod"/>.
        /// </summary>
        /// <param name="mod">A value representing the divisor of the operation.</param>
        /// <returns>
        /// A vector with each component <see cref="Mathf.PosMod(int, int)"/> by <paramref name="mod"/>.
        /// </returns>
        public Vector3i PosMod(int mod)
        {
            Vector3i v = this;
            v.x = Mathf.PosMod(v.x, mod);
            v.y = Mathf.PosMod(v.y, mod);
            v.z = Mathf.PosMod(v.z, mod);
            return v;
        }

        /// <summary>
        /// Returns a vector composed of the <see cref="Mathf.PosMod(int, int)"/> of this vector's components
        /// and <paramref name="modv"/>'s components.
        /// </summary>
        /// <param name="modv">A vector representing the divisors of the operation.</param>
        /// <returns>
        /// A vector with each component <see cref="Mathf.PosMod(int, int)"/> by <paramref name="modv"/>'s components.
        /// </returns>
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
        /// <returns>A vector with all components as either <c>1</c>, <c>-1</c>, or <c>0</c>.</returns>
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
        /// Zero vector, a vector with all components set to <c>0</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(0, 0, 0)</c>.</value>
        public static Vector3i Zero { get { return _zero; } }
        /// <summary>
        /// One vector, a vector with all components set to <c>1</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(1, 1, 1)</c>.</value>
        public static Vector3i One { get { return _one; } }

        /// <summary>
        /// Up unit vector.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(0, 1, 0)</c>.</value>
        public static Vector3i Up { get { return _up; } }
        /// <summary>
        /// Down unit vector.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(0, -1, 0)</c>.</value>
        public static Vector3i Down { get { return _down; } }
        /// <summary>
        /// Right unit vector. Represents the local direction of right,
        /// and the global direction of east.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(1, 0, 0)</c>.</value>
        public static Vector3i Right { get { return _right; } }
        /// <summary>
        /// Left unit vector. Represents the local direction of left,
        /// and the global direction of west.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(-1, 0, 0)</c>.</value>
        public static Vector3i Left { get { return _left; } }
        /// <summary>
        /// Forward unit vector. Represents the local direction of forward,
        /// and the global direction of north.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(0, 0, -1)</c>.</value>
        public static Vector3i Forward { get { return _forward; } }
        /// <summary>
        /// Back unit vector. Represents the local direction of back,
        /// and the global direction of south.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3i(0, 0, 1)</c>.</value>
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

        /// <summary>
        /// Adds each component of the <see cref="Vector3i"/>
        /// with the components of the given <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector3i operator +(Vector3i left, Vector3i right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector3i"/>
        /// by the components of the given <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector3i operator -(Vector3i left, Vector3i right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector3i"/>.
        /// This is the same as writing <c>new Vector3i(-v.x, -v.y, -v.z)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector3i operator -(Vector3i vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3i operator *(Vector3i vec, int scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3i operator *(int scale, Vector3i vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3i"/>
        /// by the components of the given <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3i operator *(Vector3i left, Vector3i right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3i operator /(Vector3i vec, int divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector3i"/>
        /// by the components of the given <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3i operator /(Vector3i vec, Vector3i divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3i"/>
        /// with the components of the given <see langword="int"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(int)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3i(10, -20, 30) % 7); // Prints "(3, -6, 2)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3i operator %(Vector3i vec, int divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            vec.z %= divisor;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3i"/>
        /// with the components of the given <see cref="Vector3i"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(Vector3i)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3i(10, -20, 30) % new Vector3i(7, 8, 9)); // Prints "(3, -4, 3)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3i operator %(Vector3i vec, Vector3i divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            vec.z %= divisorv.z;
            return vec;
        }

        /// <summary>
        /// Performs a bitwise AND operation with this <see cref="Vector3i"/>
        /// and the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The vector to AND with.</param>
        /// <param name="and">The integer to AND with.</param>
        /// <returns>The result of the bitwise AND.</returns>
        public static Vector3i operator &(Vector3i vec, int and)
        {
            vec.x &= and;
            vec.y &= and;
            vec.z &= and;
            return vec;
        }

        /// <summary>
        /// Performs a bitwise AND operation with this <see cref="Vector3i"/>
        /// and the given <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="vec">The left vector to AND with.</param>
        /// <param name="andv">The right vector to AND with.</param>
        /// <returns>The result of the bitwise AND.</returns>
        public static Vector3i operator &(Vector3i vec, Vector3i andv)
        {
            vec.x &= andv.x;
            vec.y &= andv.y;
            vec.z &= andv.z;
            return vec;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public static bool operator ==(Vector3i left, Vector3i right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are not equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are not equal.</returns>
        public static bool operator !=(Vector3i left, Vector3i right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector3i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector3i left, Vector3i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z < right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector3i left, Vector3i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z > right.z;
                }
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector3i left, Vector3i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z <= right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector3i left, Vector3i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z >= right.z;
                }
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Converts this <see cref="Vector3i"/> to a <see cref="Vector3"/>.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static implicit operator Vector3(Vector3i value)
        {
            return new Vector3(value.x, value.y, value.z);
        }

        /// <summary>
        /// Converts a <see cref="Vector3"/> to a <see cref="Vector3i"/>.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static explicit operator Vector3i(Vector3 value)
        {
            return new Vector3i(value);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vector is equal
        /// to the given object (<see paramref="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the vector and the object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Vector3i)
            {
                return Equals((Vector3i)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public bool Equals(Vector3i other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Vector3i"/>.
        /// </summary>
        /// <returns>A hash code for this vector.</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Vector3i"/> to a string.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z})";
        }

        /// <summary>
        /// Converts this <see cref="Vector3i"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)})";
        }
    }
}
