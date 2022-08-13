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
    /// 4-element structure that can be used to represent positions in 4D space or any other pair of numeric values.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4 : IEquatable<Vector4>
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
        public real_t x;

        /// <summary>
        /// The vector's Y component. Also accessible by using the index position <c>[1]</c>.
        /// </summary>
        public real_t y;

        /// <summary>
        /// The vector's Z component. Also accessible by using the index position <c>[2]</c>.
        /// </summary>
        public real_t z;

        /// <summary>
        /// The vector's W component. Also accessible by using the index position <c>[3]</c>.
        /// </summary>
        public real_t w;

        /// <summary>
        /// Access vector components using their index.
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
        public real_t this[int index]
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
        public void Deconstruct(out real_t x, out real_t y, out real_t z, out real_t w)
        {
            x = this.x;
            y = this.y;
            z = this.z;
            w = this.w;
        }

        internal void Normalize()
        {
            real_t lengthsq = LengthSquared();

            if (lengthsq == 0)
            {
                x = y = z = w = 0f;
            }
            else
            {
                real_t length = Mathf.Sqrt(lengthsq);
                x /= length;
                y /= length;
                z /= length;
                w /= length;
            }
        }

        /// <summary>
        /// Returns a new vector with all components in absolute values (i.e. positive).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Abs(real_t)"/> called on each component.</returns>
        public Vector4 Abs()
        {
            return new Vector4(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z), Mathf.Abs(w));
        }

        /// <summary>
        /// Returns a new vector with all components rounded up (towards positive infinity).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Ceil"/> called on each component.</returns>
        public Vector4 Ceil()
        {
            return new Vector4(Mathf.Ceil(x), Mathf.Ceil(y), Mathf.Ceil(z), Mathf.Ceil(w));
        }

        /// <summary>
        /// Returns a new vector with all components clamped between the
        /// components of <paramref name="min"/> and <paramref name="max"/> using
        /// <see cref="Mathf.Clamp(real_t, real_t, real_t)"/>.
        /// </summary>
        /// <param name="min">The vector with minimum allowed values.</param>
        /// <param name="max">The vector with maximum allowed values.</param>
        /// <returns>The vector with all components clamped.</returns>
        public Vector4 Clamp(Vector4 min, Vector4 max)
        {
            return new Vector4
            (
                Mathf.Clamp(x, min.x, max.x),
                Mathf.Clamp(y, min.y, max.y),
                Mathf.Clamp(z, min.z, max.z),
                Mathf.Clamp(w, min.w, max.w)
            );
        }

        /// <summary>
        /// Performs a cubic interpolation between vectors <paramref name="preA"/>, this vector,
        /// <paramref name="b"/>, and <paramref name="postB"/>, by the given amount <paramref name="weight"/>.
        /// </summary>
        /// <param name="b">The destination vector.</param>
        /// <param name="preA">A vector before this vector.</param>
        /// <param name="postB">A vector after <paramref name="b"/>.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The interpolated vector.</returns>
        public Vector4 CubicInterpolate(Vector4 b, Vector4 preA, Vector4 postB, real_t weight)
        {
            return new Vector4
            (
                Mathf.CubicInterpolate(x, b.x, preA.x, postB.x, weight),
                Mathf.CubicInterpolate(y, b.y, preA.y, postB.y, weight),
                Mathf.CubicInterpolate(y, b.z, preA.z, postB.z, weight),
                Mathf.CubicInterpolate(w, b.w, preA.w, postB.w, weight)
            );
        }

        /// <summary>
        /// Returns the normalized vector pointing from this vector to <paramref name="to"/>.
        /// </summary>
        /// <param name="to">The other vector to point towards.</param>
        /// <returns>The direction from this vector to <paramref name="to"/>.</returns>
        public Vector4 DirectionTo(Vector4 to)
        {
            Vector4 ret = new Vector4(to.x - x, to.y - y, to.z - z, to.w - w);
            ret.Normalize();
            return ret;
        }

        /// <summary>
        /// Returns the squared distance between this vector and <paramref name="to"/>.
        /// This method runs faster than <see cref="DistanceTo"/>, so prefer it if
        /// you need to compare vectors or need the squared distance for some formula.
        /// </summary>
        /// <param name="to">The other vector to use.</param>
        /// <returns>The squared distance between the two vectors.</returns>
        public real_t DistanceSquaredTo(Vector4 to)
        {
            return (to - this).LengthSquared();
        }

        /// <summary>
        /// Returns the distance between this vector and <paramref name="to"/>.
        /// </summary>
        /// <param name="to">The other vector to use.</param>
        /// <returns>The distance between the two vectors.</returns>
        public real_t DistanceTo(Vector4 to)
        {
            return (to - this).Length();
        }

        /// <summary>
        /// Returns the dot product of this vector and <paramref name="with"/>.
        /// </summary>
        /// <param name="with">The other vector to use.</param>
        /// <returns>The dot product of the two vectors.</returns>
        public real_t Dot(Vector4 with)
        {
            return (x * with.x) + (y * with.y) + (z * with.z) + (w + with.w);
        }

        /// <summary>
        /// Returns a new vector with all components rounded down (towards negative infinity).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Floor"/> called on each component.</returns>
        public Vector4 Floor()
        {
            return new Vector4(Mathf.Floor(x), Mathf.Floor(y), Mathf.Floor(z), Mathf.Floor(w));
        }

        /// <summary>
        /// Returns the inverse of this vector. This is the same as <c>new Vector4(1 / v.x, 1 / v.y, 1 / v.z, 1 / v.w)</c>.
        /// </summary>
        /// <returns>The inverse of this vector.</returns>
        public Vector4 Inverse()
        {
            return new Vector4(1 / x, 1 / y, 1 / z, 1 / w);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vector is normalized, and <see langword="false"/> otherwise.
        /// </summary>
        /// <returns>A <see langword="bool"/> indicating whether or not the vector is normalized.</returns>
        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared() - 1.0f) < Mathf.Epsilon;
        }

        /// <summary>
        /// Returns the length (magnitude) of this vector.
        /// </summary>
        /// <seealso cref="LengthSquared"/>
        /// <returns>The length of this vector.</returns>
        public real_t Length()
        {
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;
            real_t w2 = w * w;

            return Mathf.Sqrt(x2 + y2 + z2 + w2);
        }

        /// <summary>
        /// Returns the squared length (squared magnitude) of this vector.
        /// This method runs faster than <see cref="Length"/>, so prefer it if
        /// you need to compare vectors or need the squared length for some formula.
        /// </summary>
        /// <returns>The squared length of this vector.</returns>
        public real_t LengthSquared()
        {
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;
            real_t w2 = w * w;

            return x2 + y2 + z2 + w2;
        }

        /// <summary>
        /// Returns the result of the linear interpolation between
        /// this vector and <paramref name="to"/> by amount <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">The destination vector for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting vector of the interpolation.</returns>
        public Vector4 Lerp(Vector4 to, real_t weight)
        {
            return new Vector4
            (
                Mathf.Lerp(x, to.x, weight),
                Mathf.Lerp(y, to.y, weight),
                Mathf.Lerp(z, to.z, weight),
                Mathf.Lerp(w, to.w, weight)
            );
        }

        /// <summary>
        /// Returns the axis of the vector's highest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.X"/>.
        /// </summary>
        /// <returns>The index of the highest axis.</returns>
        public Axis MaxAxisIndex()
        {
            int max_index = 0;
            real_t max_value = x;
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
            real_t min_value = x;
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
        /// Returns the vector scaled to unit length. Equivalent to <c>v / v.Length()</c>.
        /// </summary>
        /// <returns>A normalized version of the vector.</returns>
        public Vector4 Normalized()
        {
            Vector4 v = this;
            v.Normalize();
            return v;
        }

        /// <summary>
        /// Returns a vector composed of the <see cref="Mathf.PosMod(real_t, real_t)"/> of this vector's components
        /// and <paramref name="mod"/>.
        /// </summary>
        /// <param name="mod">A value representing the divisor of the operation.</param>
        /// <returns>
        /// A vector with each component <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="mod"/>.
        /// </returns>
        public Vector4 PosMod(real_t mod)
        {
            return new Vector4(
                Mathf.PosMod(x, mod),
                Mathf.PosMod(y, mod),
                Mathf.PosMod(z, mod),
                Mathf.PosMod(w, mod)
            );
        }

        /// <summary>
        /// Returns a vector composed of the <see cref="Mathf.PosMod(real_t, real_t)"/> of this vector's components
        /// and <paramref name="modv"/>'s components.
        /// </summary>
        /// <param name="modv">A vector representing the divisors of the operation.</param>
        /// <returns>
        /// A vector with each component <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="modv"/>'s components.
        /// </returns>
        public Vector4 PosMod(Vector4 modv)
        {
            return new Vector4(
                Mathf.PosMod(x, modv.x),
                Mathf.PosMod(y, modv.y),
                Mathf.PosMod(z, modv.z),
                Mathf.PosMod(w, modv.w)
            );
        }

        /// <summary>
        /// Returns this vector with all components rounded to the nearest integer,
        /// with halfway cases rounded towards the nearest multiple of two.
        /// </summary>
        /// <returns>The rounded vector.</returns>
        public Vector4 Round()
        {
            return new Vector4(Mathf.Round(x), Mathf.Round(y), Mathf.Round(z), Mathf.Round(w));
        }

        /// <summary>
        /// Returns a vector with each component set to one or negative one, depending
        /// on the signs of this vector's components, or zero if the component is zero,
        /// by calling <see cref="Mathf.Sign(real_t)"/> on each component.
        /// </summary>
        /// <returns>A vector with all components as either <c>1</c>, <c>-1</c>, or <c>0</c>.</returns>
        public Vector4 Sign()
        {
            Vector4 v;
            v.x = Mathf.Sign(x);
            v.y = Mathf.Sign(y);
            v.z = Mathf.Sign(z);
            v.w = Mathf.Sign(w);
            return v;
        }

        /// <summary>
        /// Returns this vector with each component snapped to the nearest multiple of <paramref name="step"/>.
        /// This can also be used to round to an arbitrary number of decimals.
        /// </summary>
        /// <param name="step">A vector value representing the step size to snap to.</param>
        public Vector4 Snapped(Vector4 step)
        {
            return new Vector4(
                Mathf.Snapped(x, step.x),
                Mathf.Snapped(y, step.y),
                Mathf.Snapped(z, step.z),
                Mathf.Snapped(w, step.w)
            );
        }

        // Constants
        private static readonly Vector4 _zero = new Vector4(0, 0, 0, 0);
        private static readonly Vector4 _one = new Vector4(1, 1, 1, 1);
        private static readonly Vector4 _inf = new Vector4(Mathf.Inf, Mathf.Inf, Mathf.Inf, Mathf.Inf);

        /// <summary>
        /// Zero vector, a vector with all components set to <c>0</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector4(0, 0, 0, 0)</c>.</value>
        public static Vector4 Zero { get { return _zero; } }
        /// <summary>
        /// One vector, a vector with all components set to <c>1</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector4(1, 1, 1, 1)</c>.</value>
        public static Vector4 One { get { return _one; } }
        /// <summary>
        /// Infinity vector, a vector with all components set to <see cref="Mathf.Inf"/>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector4(Mathf.Inf, Mathf.Inf, Mathf.Inf, Mathf.Inf)</c>.</value>
        public static Vector4 Inf { get { return _inf; } }

        /// <summary>
        /// Constructs a new <see cref="Vector4"/> with the given components.
        /// </summary>
        /// <param name="x">The vector's X component.</param>
        /// <param name="y">The vector's Y component.</param>
        /// <param name="z">The vector's Z component.</param>
        /// <param name="w">The vector's W component.</param>
        public Vector4(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        /// <summary>
        /// Constructs a new <see cref="Vector4"/> from an existing <see cref="Vector4"/>.
        /// </summary>
        /// <param name="v">The existing <see cref="Vector4"/>.</param>
        public Vector4(Vector4 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
            w = v.w;
        }

        /// <summary>
        /// Adds each component of the <see cref="Vector4"/>
        /// with the components of the given <see cref="Vector4"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector4 operator +(Vector4 left, Vector4 right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            left.w += right.w;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector4"/>
        /// by the components of the given <see cref="Vector4"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector4 operator -(Vector4 left, Vector4 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            left.w -= right.w;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector4"/>.
        /// This is the same as writing <c>new Vector4(-v.x, -v.y, -v.z, -v.w)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// With floats, the number zero can be either positive or negative.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector4 operator -(Vector4 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            vec.w = -vec.w;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4 operator *(Vector4 vec, real_t scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            vec.w *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4 operator *(real_t scale, Vector4 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            vec.w *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector4"/>
        /// by the components of the given <see cref="Vector4"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector4 operator *(Vector4 left, Vector4 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            left.w *= right.w;
            return left;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector4"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector4 operator /(Vector4 vec, real_t divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            vec.w /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector4"/>
        /// by the components of the given <see cref="Vector4"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector4 operator /(Vector4 vec, Vector4 divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            vec.w /= divisorv.w;
            return vec;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are exactly equal.</returns>
        public static bool operator ==(Vector4 left, Vector4 right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are not equal.</returns>
        public static bool operator !=(Vector4 left, Vector4 right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector4"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector4 left, Vector4 right)
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
        /// Compares two <see cref="Vector4"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector4 left, Vector4 right)
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
        /// Compares two <see cref="Vector4"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector4 left, Vector4 right)
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
        /// Compares two <see cref="Vector4"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y, Z and finally W values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector4 left, Vector4 right)
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
        /// Returns <see langword="true"/> if the vector is exactly equal
        /// to the given object (<see paramref="obj"/>).
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the vector and the object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Vector4)
            {
                return Equals((Vector4)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>Whether or not the vectors are exactly equal.</returns>
        public bool Equals(Vector4 other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this vector and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Mathf.IsEqualApprox(real_t, real_t)"/> on each component.
        /// </summary>
        /// <param name="other">The other vector to compare.</param>
        /// <returns>Whether or not the vectors are approximately equal.</returns>
        public bool IsEqualApprox(Vector4 other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z) && Mathf.IsEqualApprox(w, other.w);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Vector4"/>.
        /// </summary>
        /// <returns>A hash code for this vector.</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Vector4"/> to a string.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z}, {w})";
        }

        /// <summary>
        /// Converts this <see cref="Vector4"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)}, {w.ToString(format)})";
        }
    }
}
