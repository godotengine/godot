using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot
{
    /// <summary>
    /// 3-element structure that can be used to represent 3D grid coordinates or sets of integers.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3I : IEquatable<Vector3I>
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
            Z
        }

        /// <summary>
        /// The vector's X component. Also accessible by using the index position <c>[0]</c>.
        /// </summary>
        public int X;

        /// <summary>
        /// The vector's Y component. Also accessible by using the index position <c>[1]</c>.
        /// </summary>
        public int Y;

        /// <summary>
        /// The vector's Z component. Also accessible by using the index position <c>[2]</c>.
        /// </summary>
        public int Z;

        /// <summary>
        /// Access vector components using their <paramref name="index"/>.
        /// </summary>
        /// <exception cref="ArgumentOutOfRangeException">
        /// <paramref name="index"/> is not 0, 1 or 2.
        /// </exception>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="X"/>,
        /// <c>[1]</c> is equivalent to <see cref="Y"/>,
        /// <c>[2]</c> is equivalent to <see cref="Z"/>.
        /// </value>
        public int this[int index]
        {
            readonly get
            {
                switch (index)
                {
                    case 0:
                        return X;
                    case 1:
                        return Y;
                    case 2:
                        return Z;
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
                    case 2:
                        Z = value;
                        return;
                    default:
                        throw new ArgumentOutOfRangeException(nameof(index));
                }
            }
        }

        /// <summary>
        /// Helper method for deconstruction into a tuple.
        /// </summary>
        public readonly void Deconstruct(out int x, out int y, out int z)
        {
            x = X;
            y = Y;
            z = Z;
        }

        /// <summary>
        /// Returns a new vector with all components in absolute values (i.e. positive).
        /// </summary>
        /// <returns>A vector with <see cref="Mathf.Abs(int)"/> called on each component.</returns>
        public readonly Vector3I Abs()
        {
            return new Vector3I(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z));
        }

        /// <summary>
        /// Returns a new vector with all components clamped between the
        /// components of <paramref name="min"/> and <paramref name="max"/> using
        /// <see cref="Mathf.Clamp(int, int, int)"/>.
        /// </summary>
        /// <param name="min">The vector with minimum allowed values.</param>
        /// <param name="max">The vector with maximum allowed values.</param>
        /// <returns>The vector with all components clamped.</returns>
        public readonly Vector3I Clamp(Vector3I min, Vector3I max)
        {
            return new Vector3I
            (
                Mathf.Clamp(X, min.X, max.X),
                Mathf.Clamp(Y, min.Y, max.Y),
                Mathf.Clamp(Z, min.Z, max.Z)
            );
        }

        /// <summary>
        /// Returns a new vector with all components clamped between the
        /// <paramref name="min"/> and <paramref name="max"/> using
        /// <see cref="Mathf.Clamp(int, int, int)"/>.
        /// </summary>
        /// <param name="min">The minimum allowed value.</param>
        /// <param name="max">The maximum allowed value.</param>
        /// <returns>The vector with all components clamped.</returns>
        public readonly Vector3I Clamp(int min, int max)
        {
            return new Vector3I
            (
                Mathf.Clamp(X, min, max),
                Mathf.Clamp(Y, min, max),
                Mathf.Clamp(Z, min, max)
            );
        }

        /// <summary>
        /// Returns the squared distance between this vector and <paramref name="to"/>.
        /// This method runs faster than <see cref="DistanceTo"/>, so prefer it if
        /// you need to compare vectors or need the squared distance for some formula.
        /// </summary>
        /// <param name="to">The other vector to use.</param>
        /// <returns>The squared distance between the two vectors.</returns>
        public readonly int DistanceSquaredTo(Vector3I to)
        {
            return (to - this).LengthSquared();
        }

        /// <summary>
        /// Returns the distance between this vector and <paramref name="to"/>.
        /// </summary>
        /// <seealso cref="DistanceSquaredTo(Vector3I)"/>
        /// <param name="to">The other vector to use.</param>
        /// <returns>The distance between the two vectors.</returns>
        public readonly real_t DistanceTo(Vector3I to)
        {
            return (to - this).Length();
        }

        /// <summary>
        /// Returns the length (magnitude) of this vector.
        /// </summary>
        /// <seealso cref="LengthSquared"/>
        /// <returns>The length of this vector.</returns>
        public readonly real_t Length()
        {
            int x2 = X * X;
            int y2 = Y * Y;
            int z2 = Z * Z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        /// <summary>
        /// Returns the squared length (squared magnitude) of this vector.
        /// This method runs faster than <see cref="Length"/>, so prefer it if
        /// you need to compare vectors or need the squared length for some formula.
        /// </summary>
        /// <returns>The squared length of this vector.</returns>
        public readonly int LengthSquared()
        {
            int x2 = X * X;
            int y2 = Y * Y;
            int z2 = Z * Z;

            return x2 + y2 + z2;
        }

        /// <summary>
        /// Returns the result of the component-wise maximum between
        /// this vector and <paramref name="with"/>.
        /// Equivalent to <c>new Vector3I(Mathf.Max(X, with.X), Mathf.Max(Y, with.Y), Mathf.Max(Z, with.Z))</c>.
        /// </summary>
        /// <param name="with">The other vector to use.</param>
        /// <returns>The resulting maximum vector.</returns>
        public readonly Vector3I Max(Vector3I with)
        {
            return new Vector3I
            (
                Mathf.Max(X, with.X),
                Mathf.Max(Y, with.Y),
                Mathf.Max(Z, with.Z)
            );
        }

        /// <summary>
        /// Returns the result of the component-wise maximum between
        /// this vector and <paramref name="with"/>.
        /// Equivalent to <c>new Vector3I(Mathf.Max(X, with), Mathf.Max(Y, with), Mathf.Max(Z, with))</c>.
        /// </summary>
        /// <param name="with">The other value to use.</param>
        /// <returns>The resulting maximum vector.</returns>
        public readonly Vector3I Max(int with)
        {
            return new Vector3I
            (
                Mathf.Max(X, with),
                Mathf.Max(Y, with),
                Mathf.Max(Z, with)
            );
        }

        /// <summary>
        /// Returns the result of the component-wise minimum between
        /// this vector and <paramref name="with"/>.
        /// Equivalent to <c>new Vector3I(Mathf.Min(X, with.X), Mathf.Min(Y, with.Y), Mathf.Min(Z, with.Z))</c>.
        /// </summary>
        /// <param name="with">The other vector to use.</param>
        /// <returns>The resulting minimum vector.</returns>
        public readonly Vector3I Min(Vector3I with)
        {
            return new Vector3I
            (
                Mathf.Min(X, with.X),
                Mathf.Min(Y, with.Y),
                Mathf.Min(Z, with.Z)
            );
        }

        /// <summary>
        /// Returns the result of the component-wise minimum between
        /// this vector and <paramref name="with"/>.
        /// Equivalent to <c>new Vector3I(Mathf.Min(X, with), Mathf.Min(Y, with), Mathf.Min(Z, with))</c>.
        /// </summary>
        /// <param name="with">The other value to use.</param>
        /// <returns>The resulting minimum vector.</returns>
        public readonly Vector3I Min(int with)
        {
            return new Vector3I
            (
                Mathf.Min(X, with),
                Mathf.Min(Y, with),
                Mathf.Min(Z, with)
            );
        }

        /// <summary>
        /// Returns the axis of the vector's highest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.X"/>.
        /// </summary>
        /// <returns>The index of the highest axis.</returns>
        public readonly Axis MaxAxisIndex()
        {
            return X < Y ? (Y < Z ? Axis.Z : Axis.Y) : (X < Z ? Axis.Z : Axis.X);
        }

        /// <summary>
        /// Returns the axis of the vector's lowest value. See <see cref="Axis"/>.
        /// If all components are equal, this method returns <see cref="Axis.Z"/>.
        /// </summary>
        /// <returns>The index of the lowest axis.</returns>
        public readonly Axis MinAxisIndex()
        {
            return X < Y ? (X < Z ? Axis.X : Axis.Z) : (Y < Z ? Axis.Y : Axis.Z);
        }

        /// <summary>
        /// Returns a vector with each component set to one or negative one, depending
        /// on the signs of this vector's components, or zero if the component is zero,
        /// by calling <see cref="Mathf.Sign(int)"/> on each component.
        /// </summary>
        /// <returns>A vector with all components as either <c>1</c>, <c>-1</c>, or <c>0</c>.</returns>
        public readonly Vector3I Sign()
        {
            Vector3I v = this;
            v.X = Mathf.Sign(v.X);
            v.Y = Mathf.Sign(v.Y);
            v.Z = Mathf.Sign(v.Z);
            return v;
        }

        /// <summary>
        /// Returns a new vector with each component snapped to the closest multiple of the corresponding component in <paramref name="step"/>.
        /// </summary>
        /// <param name="step">A vector value representing the step size to snap to.</param>
        /// <returns>The snapped vector.</returns>
        public readonly Vector3I Snapped(Vector3I step)
        {
            return new Vector3I
            (
                (int)Mathf.Snapped((double)X, (double)step.X),
                (int)Mathf.Snapped((double)Y, (double)step.Y),
                (int)Mathf.Snapped((double)Z, (double)step.Z)
            );
        }

        /// <summary>
        /// Returns a new vector with each component snapped to the closest multiple of <paramref name="step"/>.
        /// </summary>
        /// <param name="step">The step size to snap to.</param>
        /// <returns>The snapped vector.</returns>
        public readonly Vector3I Snapped(int step)
        {
            return new Vector3I
            (
                (int)Mathf.Snapped((double)X, (double)step),
                (int)Mathf.Snapped((double)Y, (double)step),
                (int)Mathf.Snapped((double)Z, (double)step)
            );
        }

        // Constants
        private static readonly Vector3I _minValue = new Vector3I(int.MinValue, int.MinValue, int.MinValue);
        private static readonly Vector3I _maxValue = new Vector3I(int.MaxValue, int.MaxValue, int.MaxValue);

        private static readonly Vector3I _zero = new Vector3I(0, 0, 0);
        private static readonly Vector3I _one = new Vector3I(1, 1, 1);

        private static readonly Vector3I _up = new Vector3I(0, 1, 0);
        private static readonly Vector3I _down = new Vector3I(0, -1, 0);
        private static readonly Vector3I _right = new Vector3I(1, 0, 0);
        private static readonly Vector3I _left = new Vector3I(-1, 0, 0);
        private static readonly Vector3I _forward = new Vector3I(0, 0, -1);
        private static readonly Vector3I _back = new Vector3I(0, 0, 1);

        /// <summary>
        /// Min vector, a vector with all components equal to <see cref="int.MinValue"/>. Can be used as a negative integer equivalent of <see cref="Vector3.Inf"/>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(int.MinValue, int.MinValue, int.MinValue)</c>.</value>
        public static Vector3I MinValue { get { return _minValue; } }
        /// <summary>
        /// Max vector, a vector with all components equal to <see cref="int.MaxValue"/>. Can be used as an integer equivalent of <see cref="Vector3.Inf"/>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(int.MaxValue, int.MaxValue, int.MaxValue)</c>.</value>
        public static Vector3I MaxValue { get { return _maxValue; } }

        /// <summary>
        /// Zero vector, a vector with all components set to <c>0</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(0, 0, 0)</c>.</value>
        public static Vector3I Zero { get { return _zero; } }
        /// <summary>
        /// One vector, a vector with all components set to <c>1</c>.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(1, 1, 1)</c>.</value>
        public static Vector3I One { get { return _one; } }

        /// <summary>
        /// Up unit vector.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(0, 1, 0)</c>.</value>
        public static Vector3I Up { get { return _up; } }
        /// <summary>
        /// Down unit vector.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(0, -1, 0)</c>.</value>
        public static Vector3I Down { get { return _down; } }
        /// <summary>
        /// Right unit vector. Represents the local direction of right,
        /// and the global direction of east.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(1, 0, 0)</c>.</value>
        public static Vector3I Right { get { return _right; } }
        /// <summary>
        /// Left unit vector. Represents the local direction of left,
        /// and the global direction of west.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(-1, 0, 0)</c>.</value>
        public static Vector3I Left { get { return _left; } }
        /// <summary>
        /// Forward unit vector. Represents the local direction of forward,
        /// and the global direction of north.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(0, 0, -1)</c>.</value>
        public static Vector3I Forward { get { return _forward; } }
        /// <summary>
        /// Back unit vector. Represents the local direction of back,
        /// and the global direction of south.
        /// </summary>
        /// <value>Equivalent to <c>new Vector3I(0, 0, 1)</c>.</value>
        public static Vector3I Back { get { return _back; } }

        /// <summary>
        /// Constructs a new <see cref="Vector3I"/> with the given components.
        /// </summary>
        /// <param name="x">The vector's X component.</param>
        /// <param name="y">The vector's Y component.</param>
        /// <param name="z">The vector's Z component.</param>
        public Vector3I(int x, int y, int z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        /// <summary>
        /// Adds each component of the <see cref="Vector3I"/>
        /// with the components of the given <see cref="Vector3I"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector3I operator +(Vector3I left, Vector3I right)
        {
            left.X += right.X;
            left.Y += right.Y;
            left.Z += right.Z;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector3I"/>
        /// by the components of the given <see cref="Vector3I"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector3I operator -(Vector3I left, Vector3I right)
        {
            left.X -= right.X;
            left.Y -= right.Y;
            left.Z -= right.Z;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector3I"/>.
        /// This is the same as writing <c>new Vector3I(-v.X, -v.Y, -v.Z)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector3I operator -(Vector3I vec)
        {
            vec.X = -vec.X;
            vec.Y = -vec.Y;
            vec.Z = -vec.Z;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3I"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3I operator *(Vector3I vec, int scale)
        {
            vec.X *= scale;
            vec.Y *= scale;
            vec.Z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3I"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3I operator *(int scale, Vector3I vec)
        {
            vec.X *= scale;
            vec.Y *= scale;
            vec.Z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3I"/>
        /// by the components of the given <see cref="Vector3I"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3I operator *(Vector3I left, Vector3I right)
        {
            left.X *= right.X;
            left.Y *= right.Y;
            left.Z *= right.Z;
            return left;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector3I"/>
        /// by the given <see langword="int"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3I operator /(Vector3I vec, int divisor)
        {
            vec.X /= divisor;
            vec.Y /= divisor;
            vec.Z /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector3I"/>
        /// by the components of the given <see cref="Vector3I"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3I operator /(Vector3I vec, Vector3I divisorv)
        {
            vec.X /= divisorv.X;
            vec.Y /= divisorv.Y;
            vec.Z /= divisorv.Z;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3I"/>
        /// with the components of the given <see langword="int"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="Mathf.PosMod(int, int)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3I(10, -20, 30) % 7); // Prints "(3, -6, 2)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3I operator %(Vector3I vec, int divisor)
        {
            vec.X %= divisor;
            vec.Y %= divisor;
            vec.Z %= divisor;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3I"/>
        /// with the components of the given <see cref="Vector3I"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="Mathf.PosMod(int, int)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3I(10, -20, 30) % new Vector3I(7, 8, 9)); // Prints "(3, -4, 3)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3I operator %(Vector3I vec, Vector3I divisorv)
        {
            vec.X %= divisorv.X;
            vec.Y %= divisorv.Y;
            vec.Z %= divisorv.Z;
            return vec;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public static bool operator ==(Vector3I left, Vector3I right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are not equal.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are not equal.</returns>
        public static bool operator !=(Vector3I left, Vector3I right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector3I"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector3I left, Vector3I right)
        {
            if (left.X == right.X)
            {
                if (left.Y == right.Y)
                {
                    return left.Z < right.Z;
                }
                return left.Y < right.Y;
            }
            return left.X < right.X;
        }

        /// <summary>
        /// Compares two <see cref="Vector3I"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector3I left, Vector3I right)
        {
            if (left.X == right.X)
            {
                if (left.Y == right.Y)
                {
                    return left.Z > right.Z;
                }
                return left.Y > right.Y;
            }
            return left.X > right.X;
        }

        /// <summary>
        /// Compares two <see cref="Vector3I"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector3I left, Vector3I right)
        {
            if (left.X == right.X)
            {
                if (left.Y == right.Y)
                {
                    return left.Z <= right.Z;
                }
                return left.Y < right.Y;
            }
            return left.X < right.X;
        }

        /// <summary>
        /// Compares two <see cref="Vector3I"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector3I left, Vector3I right)
        {
            if (left.X == right.X)
            {
                if (left.Y == right.Y)
                {
                    return left.Z >= right.Z;
                }
                return left.Y > right.Y;
            }
            return left.X > right.X;
        }

        /// <summary>
        /// Converts this <see cref="Vector3I"/> to a <see cref="Vector3"/>.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static implicit operator Vector3(Vector3I value)
        {
            return new Vector3(value.X, value.Y, value.Z);
        }

        /// <summary>
        /// Converts a <see cref="Vector3"/> to a <see cref="Vector3I"/> by truncating
        /// components' fractional parts (rounding towards zero). For a different
        /// behavior consider passing the result of <see cref="Vector3.Ceil"/>,
        /// <see cref="Vector3.Floor"/> or <see cref="Vector3.Round"/> to this conversion operator instead.
        /// </summary>
        /// <param name="value">The vector to convert.</param>
        public static explicit operator Vector3I(Vector3 value)
        {
            return new Vector3I((int)value.X, (int)value.Y, (int)value.Z);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vector is equal
        /// to the given object (<paramref name="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the vector and the object are equal.</returns>
        public override readonly bool Equals([NotNullWhen(true)] object? obj)
        {
            return obj is Vector3I other && Equals(other);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are equal.
        /// </summary>
        /// <param name="other">The other vector.</param>
        /// <returns>Whether or not the vectors are equal.</returns>
        public readonly bool Equals(Vector3I other)
        {
            return X == other.X && Y == other.Y && Z == other.Z;
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Vector3I"/>.
        /// </summary>
        /// <returns>A hash code for this vector.</returns>
        public override readonly int GetHashCode()
        {
            return HashCode.Combine(X, Y, Z);
        }

        /// <summary>
        /// Converts this <see cref="Vector3I"/> to a string.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public override readonly string ToString() => ToString(null);

        /// <summary>
        /// Converts this <see cref="Vector3I"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this vector.</returns>
        public readonly string ToString(string? format)
        {
            return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)})";
        }
    }
}
