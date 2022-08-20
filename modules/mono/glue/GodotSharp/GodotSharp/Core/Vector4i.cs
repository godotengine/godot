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
    /// 4-element structure that can be used to represent 4D grid coordinates or sets of integers.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4i : IEquatable<Vector4i>
    {
        /// <summary>
        /// Enumerated index values for the axes.
        /// Returned by <see cref="MaxAxisIndex"/> and <see cref="MinAxisIndex"/>.
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
            Z,
            /// <summary>
            /// The vector's W axis.
            /// </summary>
            W
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
        /// The vector's W component. Also accessible by using the index position <c>[3]</c>.
        /// </summary>
        public int w;

        /// <summary>
        /// Access vector components using their <paramref name="index"/>.
        /// </summary>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the given the <paramref name="index"/> is not 0, 1, 2 or 3.
        /// </exception>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="x"/>,
        /// <c>[1]</c> is equivalent to <see cref="y"/>,
        /// <c>[2]</c> is equivalent to <see cref="z"/>.
        /// <c>[3]</c> is equivalent to <see cref="w"/>.
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
                    case 3:
                        return w;
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
                    case 3:
                        w = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Helper method for deconstruction into a tuple.
        /// </summary>
        public void Deconstruct(out int x, out int y, out int z, out int w)
        {
            x = this.x;
            y = this.y;
            z = this.z;
            w = this.w;
        }

        /// <summary>
        /// Returns a new vector with all components in absolute values (i.e. positive).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Abs(int)"/> called on each component.</returns>
        public Vector4i Abs()
        {
            return new Vector4i(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z), Mathf.Abs(w));
        }

        /// <summary>
        /// Returns a new vector with all components clamped between the
        /// components of <paramref name="min"/> and <paramref name="max"/> using
        /// <see cref="Mathf.Clamp(int, int, int)"/>.
        /// </summary>
        /// <param name="min">The vector with minimum allowed values.</param>
        /// <param name="max">The vector with maximum allowed values.</param>
        /// <returns>The vector with all components clamped.</returns>
        public Vector4i Clamp(Vector4i min, Vector4i max)
        {
            return new Vector4i
            (
                Mathf.Clamp(x, min.x, max.x),
                Mathf.Clamp(y, min.y, max.y),
                Mathf.Clamp(z, min.z, max.z),
                Mathf.Clamp(w, min.w, max.w)
            );
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
            int w2 = w * w;

            return Mathf.Sqrt(x2 + y2 + z2 + w2);
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
            int w2 = w * w;

            return x2 + y2 + z2 + w2;
        }

        /// <summary>
        /// Returns the axis of the vector's highest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.X"/>.
        /// </summary>
        /// <returns>The index of the highest axis.</returns>
        public Axis MaxAxisIndex()
        {
            int max_index = 0;
            int max_value = x;
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

        /// <summary>
        /// Returns the axis of the vector's lowest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.W"/>.
        /// </summary>
        /// <returns>The index of the lowest axis.</returns>
        public Axis MinAxisIndex()
        {
            int min_index = 0;
            int min_value = x;
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

        /// <summary>
        /// Returns a vector with each component set to one or negative one, depending
        /// on the signs of this vector's components, or zero if the component is zero,
        /// by calling <see cref="Mathf.Sign(int)"/> on each component.
        /// </summary>
        /// <returns>A vector with all components as either <c>1</c>, <c>-1</c>, or <c>0</c>.</returns>
        public Vector4i Sign()
        {
            return new Vector4i(Mathf.Sign(x), Mathf.Sign(y), Mathf.Sign(z), Mathf.Sign(w));
        }

        // Constants
        private static readonly Vector4i _zero = new Vector4i(0, 0, 0, 0);
        private static readonly Vector4i _one = new Vector4i(1, 1, 1, 1);

        /// <summary>
        /// Zero vector, a vector with all components set to <c>0</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector4i(0, 0, 0, 0)</c>.</value>
        public static Vector4i Zero { get { return _zero; } }
        /// <summary>
        /// One vector, a vector with all components set to <c>1</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector4i(1, 1, 1, 1)</c>.</value>
        public static Vector4i One { get { return _one; } }

        /// <summary>
        /// Constructs a new <see cref="Vector4i"/> with the given components.
        /// </summary>
        /// <param name="x">The vector's X component.</param>
        /// <param name="y">The vector's Y component.</param>
        /// <param name="z">The vector's Z component.</param>
        /// <param name="w">The vector's W component.</param>
        public Vector4i(int x, int y, int z, int w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        /// <summary>
        /// Constructs a new <see cref="Vector4i"/> from an existing <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="vi">The existing <see cref="Vector4i"/>.</param>
        public Vector4i(Vector4i vi)
        {
            this.x = vi.x;
            this.y = vi.y;
            this.z = vi.z;
            this.w = vi.w;
        }

        /// <summary>
        /// Constructs a new <see cref="Vector4i"/> from an existing <see cref="Vector4"/>
        /// by rounding the components via <see cref="Mathf.RoundToInt(real_t)"/>.
        /// </summary>
        /// <param name="v">The <see cref="Vector4"/> to convert.</param>
        public Vector4i(Vector4 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
            this.z = Mathf.RoundToInt(v.z);
            this.w = Mathf.RoundToInt(v.w);
        }

        /// <summary>
        /// Adds each component of the <see cref="Vector4i"/>
        /// with the components of the given <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector4i operator +(Vector4i left, Vector4i right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            left.w += right.w;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector4i"/>
        /// by the components of the given <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector4i operator -(Vector4i left, Vector4i right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            left.w -= right.w;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector4i"/>.
        /// This is the same as writing <c>new Vector4i(-v.x, -v.y, -v.z, -v.w)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector4i operator -(Vector4i vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            vec.w = -vec.w;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4i operator *(Vector4i vec, int scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            vec.w *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4i operator *(int scale, Vector4i vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            vec.w *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4i"/>
        /// by the components of the given <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4i operator *(Vector4i left, Vector4i right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            left.w *= right.w;
            return left;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector4i"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector4i operator /(Vector4i vec, int divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            vec.w /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector4i"/>
        /// by the components of the given <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector4i operator /(Vector4i vec, Vector4i divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            vec.w /= divisorv.w;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector4i"/>
        /// with the components of the given <see langword="int"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vecto43i(10, -20, 30, -40) % 7); // Prints "(3, -6, 2, -5)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector4i operator %(Vector4i vec, int divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            vec.z %= divisor;
            vec.w %= divisor;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector4i"/>
        /// with the components of the given <see cref="Vector4i"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector4i(10, -20, 30, -40) % new Vector4i(6, 7, 8, 9)); // Prints "(4, -6, 6, -4)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector4i operator %(Vector4i vec, Vector4i divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            vec.z %= divisorv.z;
            vec.w %= divisorv.w;
            return vec;
        }

        /// <summary>
        /// Performs a bitwise AND operation with this <see cref="Vector4i"/>
        /// and the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The vector to AND with.</param>
        /// <param name="and">The integer to AND with.</param>
        /// <returns>The result of the bitwise AND.</returns>
        public static Vector4i operator &(Vector4i vec, int and)
        {
            vec.x &= and;
            vec.y &= and;
            vec.z &= and;
            vec.w &= and;
            return vec;
        }

        /// <summary>
        /// Performs a bitwise AND operation with this <see cref="Vector4i"/>
        /// and the given <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="vec">The left vector to AND with.</param>
        /// <param name="andv">The right vector to AND with.</param>
        /// <returns>The result of the bitwise AND.</returns>
        public static Vector4i operator &(Vector4i vec, Vector4i andv)
        {
            vec.x &= andv.x;
            vec.y &= andv.y;
            vec.z &= andv.z;
            vec.w &= andv.w;
            return vec;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public static bool operator ==(Vector4i left, Vector4i right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are not equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are not equal.</returns>
        public static bool operator !=(Vector4i left, Vector4i right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector4i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector4i left, Vector4i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    if (left.z == right.z)
                    {
                        return left.w < right.w;
                    }
                    return left.z < right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector4i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector4i left, Vector4i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    if (left.z == right.z)
                    {
                        return left.w > right.w;
                    }
                    return left.z > right.z;
                }
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector4i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector4i left, Vector4i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    if (left.z == right.z)
                    {
                        return left.w <= right.w;
                    }
                    return left.z < right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector4i"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector4i left, Vector4i right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    if (left.z == right.z)
                    {
                        return left.w >= right.w;
                    }
                    return left.z > right.z;
                }
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Converts this <see cref="Vector4i"/> to a <see cref="Vector4"/>.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static implicit operator Vector4(Vector4i value)
        {
            return new Vector4(value.x, value.y, value.z, value.w);
        }

        /// <summary>
        /// Converts a <see cref="Vector4"/> to a <see cref="Vector4i"/>.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static explicit operator Vector4i(Vector4 value)
        {
            return new Vector4i(value);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vector is equal
        /// to the given object (<see paramref="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the vector and the object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Vector4i)
            {
                return Equals((Vector4i)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public bool Equals(Vector4i other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Vector4i"/>.
        /// </summary>
        /// <returns>A hash code for this vector.</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Vector4i"/> to a string.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z}, {w})";
        }

        /// <summary>
        /// Converts this <see cref="Vector4i"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)}), {w.ToString(format)})";
        }
    }
}
