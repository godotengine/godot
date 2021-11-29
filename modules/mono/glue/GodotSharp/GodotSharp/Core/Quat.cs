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
    /// A unit quaternion used for representing 3D rotations.
    /// Quaternions need to be normalized to be used for rotation.
    ///
    /// It is similar to <see cref="Basis"/>, which implements matrix
    /// representation of rotations, and can be parametrized using both
    /// an axis-angle pair or Euler angles. Basis stores rotation, scale,
    /// and shearing, while Quat only stores rotation.
    ///
    /// Due to its compactness and the way it is stored in memory, certain
    /// operations (obtaining axis-angle and performing SLERP, in particular)
    /// are more efficient and robust against floating-point errors.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Quat : IEquatable<Quat>
    {
        /// <summary>
        /// X component of the quaternion (imaginary <c>i</c> axis part).
        /// Quaternion components should usually not be manipulated directly.
        /// </summary>
        public real_t x;

        /// <summary>
        /// Y component of the quaternion (imaginary <c>j</c> axis part).
        /// Quaternion components should usually not be manipulated directly.
        /// </summary>
        public real_t y;

        /// <summary>
        /// Z component of the quaternion (imaginary <c>k</c> axis part).
        /// Quaternion components should usually not be manipulated directly.
        /// </summary>
        public real_t z;

        /// <summary>
        /// W component of the quaternion (real part).
        /// Quaternion components should usually not be manipulated directly.
        /// </summary>
        public real_t w;

        /// <summary>
        /// Access quaternion components using their index.
        /// </summary>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="x"/>,
        /// <c>[1]</c> is equivalent to <see cref="y"/>,
        /// <c>[2]</c> is equivalent to <see cref="z"/>,
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
                        break;
                    case 1:
                        y = value;
                        break;
                    case 2:
                        z = value;
                        break;
                    case 3:
                        w = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Returns the length (magnitude) of the quaternion.
        /// </summary>
        /// <seealso cref="LengthSquared"/>
        /// <value>Equivalent to <c>Mathf.Sqrt(LengthSquared)</c>.</value>
        public real_t Length
        {
            get { return Mathf.Sqrt(LengthSquared); }
        }

        /// <summary>
        /// Returns the squared length (squared magnitude) of the quaternion.
        /// This method runs faster than <see cref="Length"/>, so prefer it if
        /// you need to compare quaternions or need the squared length for some formula.
        /// </summary>
        /// <value>Equivalent to <c>Dot(this)</c>.</value>
        public real_t LengthSquared
        {
            get { return Dot(this); }
        }

        /// <summary>
        /// Returns the angle between this quaternion and <paramref name="to"/>.
        /// This is the magnitude of the angle you would need to rotate
        /// by to get from one to the other.
        ///
        /// Note: This method has an abnormally high amount
        /// of floating-point error, so methods such as
        /// <see cref="Mathf.IsZeroApprox"/> will not work reliably.
        /// </summary>
        /// <param name="to">The other quaternion.</param>
        /// <returns>The angle between the quaternions.</returns>
        public real_t AngleTo(Quat to)
        {
            real_t dot = Dot(to);
            return Mathf.Acos(Mathf.Clamp(dot * dot * 2 - 1, -1, 1));
        }

        /// <summary>
        /// Performs a cubic spherical interpolation between quaternions <paramref name="preA"/>, this quaternion,
        /// <paramref name="b"/>, and <paramref name="postB"/>, by the given amount <paramref name="weight"/>.
        /// </summary>
        /// <param name="b">The destination quaternion.</param>
        /// <param name="preA">A quaternion before this quaternion.</param>
        /// <param name="postB">A quaternion after <paramref name="b"/>.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The interpolated quaternion.</returns>
        public Quat CubicSlerp(Quat b, Quat preA, Quat postB, real_t weight)
        {
            real_t t2 = (1.0f - weight) * weight * 2f;
            Quat sp = Slerp(b, weight);
            Quat sq = preA.Slerpni(postB, weight);
            return sp.Slerpni(sq, t2);
        }

        /// <summary>
        /// Returns the dot product of two quaternions.
        /// </summary>
        /// <param name="b">The other quaternion.</param>
        /// <returns>The dot product.</returns>
        public real_t Dot(Quat b)
        {
            return (x * b.x) + (y * b.y) + (z * b.z) + (w * b.w);
        }

        /// <summary>
        /// Returns Euler angles (in the YXZ convention: when decomposing,
        /// first Z, then X, and Y last) corresponding to the rotation
        /// represented by the unit quaternion. Returned vector contains
        /// the rotation angles in the format (X angle, Y angle, Z angle).
        /// </summary>
        /// <returns>The Euler angle representation of this quaternion.</returns>
        public Vector3 GetEuler()
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            var basis = new Basis(this);
            return basis.GetEuler();
        }

        /// <summary>
        /// Returns the inverse of the quaternion.
        /// </summary>
        /// <returns>The inverse quaternion.</returns>
        public Quat Inverse()
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            return new Quat(-x, -y, -z, w);
        }

        /// <summary>
        /// Returns whether the quaternion is normalized or not.
        /// </summary>
        /// <returns>A <see langword="bool"/> for whether the quaternion is normalized or not.</returns>
        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared - 1) <= Mathf.Epsilon;
        }

        /// <summary>
        /// Returns a copy of the quaternion, normalized to unit length.
        /// </summary>
        /// <returns>The normalized quaternion.</returns>
        public Quat Normalized()
        {
            return this / Length;
        }

        [Obsolete("Set is deprecated. Use the Quat(" + nameof(real_t) + ", " + nameof(real_t) + ", " + nameof(real_t) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void Set(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        [Obsolete("Set is deprecated. Use the Quat(" + nameof(Quat) + ") constructor instead.", error: true)]
        public void Set(Quat q)
        {
            this = q;
        }

        [Obsolete("SetAxisAngle is deprecated. Use the Quat(" + nameof(Vector3) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void SetAxisAngle(Vector3 axis, real_t angle)
        {
            this = new Quat(axis, angle);
        }

        [Obsolete("SetEuler is deprecated. Use the Quat(" + nameof(Vector3) + ") constructor instead.", error: true)]
        public void SetEuler(Vector3 eulerYXZ)
        {
            this = new Quat(eulerYXZ);
        }

        /// <summary>
        /// Returns the result of the spherical linear interpolation between
        /// this quaternion and <paramref name="to"/> by amount <paramref name="weight"/>.
        ///
        /// Note: Both quaternions must be normalized.
        /// </summary>
        /// <param name="to">The destination quaternion for interpolation. Must be normalized.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting quaternion of the interpolation.</returns>
        public Quat Slerp(Quat to, real_t weight)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
            if (!to.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(to));
            }
#endif

            // Calculate cosine.
            real_t cosom = x * to.x + y * to.y + z * to.z + w * to.w;

            var to1 = new Quat();

            // Adjust signs if necessary.
            if (cosom < 0.0)
            {
                cosom = -cosom;
                to1.x = -to.x;
                to1.y = -to.y;
                to1.z = -to.z;
                to1.w = -to.w;
            }
            else
            {
                to1.x = to.x;
                to1.y = to.y;
                to1.z = to.z;
                to1.w = to.w;
            }

            real_t sinom, scale0, scale1;

            // Calculate coefficients.
            if (1.0 - cosom > Mathf.Epsilon)
            {
                // Standard case (Slerp).
                real_t omega = Mathf.Acos(cosom);
                sinom = Mathf.Sin(omega);
                scale0 = Mathf.Sin((1.0f - weight) * omega) / sinom;
                scale1 = Mathf.Sin(weight * omega) / sinom;
            }
            else
            {
                // Quaternions are very close so we can do a linear interpolation.
                scale0 = 1.0f - weight;
                scale1 = weight;
            }

            // Calculate final values.
            return new Quat
            (
                (scale0 * x) + (scale1 * to1.x),
                (scale0 * y) + (scale1 * to1.y),
                (scale0 * z) + (scale1 * to1.z),
                (scale0 * w) + (scale1 * to1.w)
            );
        }

        /// <summary>
        /// Returns the result of the spherical linear interpolation between
        /// this quaternion and <paramref name="to"/> by amount <paramref name="weight"/>, but without
        /// checking if the rotation path is not bigger than 90 degrees.
        /// </summary>
        /// <param name="to">The destination quaternion for interpolation. Must be normalized.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting quaternion of the interpolation.</returns>
        public Quat Slerpni(Quat to, real_t weight)
        {
            real_t dot = Dot(to);

            if (Mathf.Abs(dot) > 0.9999f)
            {
                return this;
            }

            real_t theta = Mathf.Acos(dot);
            real_t sinT = 1.0f / Mathf.Sin(theta);
            real_t newFactor = Mathf.Sin(weight * theta) * sinT;
            real_t invFactor = Mathf.Sin((1.0f - weight) * theta) * sinT;

            return new Quat
            (
                (invFactor * x) + (newFactor * to.x),
                (invFactor * y) + (newFactor * to.y),
                (invFactor * z) + (newFactor * to.z),
                (invFactor * w) + (newFactor * to.w)
            );
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by this quaternion.
        /// </summary>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public Vector3 Xform(Vector3 v)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            var u = new Vector3(x, y, z);
            Vector3 uv = u.Cross(v);
            return v + (((uv * w) + u.Cross(uv)) * 2);
        }

        // Constants
        private static readonly Quat _identity = new Quat(0, 0, 0, 1);

        /// <summary>
        /// The identity quaternion, representing no rotation.
        /// Equivalent to an identity <see cref="Basis"/> matrix. If a vector is transformed by
        /// an identity quaternion, it will not change.
        /// </summary>
        /// <value>Equivalent to <c>new Quat(0, 0, 0, 1)</c>.</value>
        public static Quat Identity { get { return _identity; } }

        /// <summary>
        /// Constructs a <see cref="Quat"/> defined by the given values.
        /// </summary>
        /// <param name="x">X component of the quaternion (imaginary <c>i</c> axis part).</param>
        /// <param name="y">Y component of the quaternion (imaginary <c>j</c> axis part).</param>
        /// <param name="z">Z component of the quaternion (imaginary <c>k</c> axis part).</param>
        /// <param name="w">W component of the quaternion (real part).</param>
        public Quat(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        /// <summary>
        /// Constructs a <see cref="Quat"/> from the given <see cref="Quat"/>.
        /// </summary>
        /// <param name="q">The existing quaternion.</param>
        public Quat(Quat q)
        {
            this = q;
        }

        /// <summary>
        /// Constructs a <see cref="Quat"/> from the given <see cref="Basis"/>.
        /// </summary>
        /// <param name="basis">The <see cref="Basis"/> to construct from.</param>
        public Quat(Basis basis)
        {
            this = basis.Quat();
        }

        /// <summary>
        /// Constructs a <see cref="Quat"/> that will perform a rotation specified by
        /// Euler angles (in the YXZ convention: when decomposing, first Z, then X, and Y last),
        /// given in the vector format as (X angle, Y angle, Z angle).
        /// </summary>
        /// <param name="eulerYXZ">Euler angles that the quaternion will be rotated by.</param>
        public Quat(Vector3 eulerYXZ)
        {
            real_t halfA1 = eulerYXZ.y * 0.5f;
            real_t halfA2 = eulerYXZ.x * 0.5f;
            real_t halfA3 = eulerYXZ.z * 0.5f;

            // R = Y(a1).X(a2).Z(a3) convention for Euler angles.
            // Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
            // a3 is the angle of the first rotation, following the notation in this reference.

            real_t cosA1 = Mathf.Cos(halfA1);
            real_t sinA1 = Mathf.Sin(halfA1);
            real_t cosA2 = Mathf.Cos(halfA2);
            real_t sinA2 = Mathf.Sin(halfA2);
            real_t cosA3 = Mathf.Cos(halfA3);
            real_t sinA3 = Mathf.Sin(halfA3);

            x = (sinA1 * cosA2 * sinA3) + (cosA1 * sinA2 * cosA3);
            y = (sinA1 * cosA2 * cosA3) - (cosA1 * sinA2 * sinA3);
            z = (cosA1 * cosA2 * sinA3) - (sinA1 * sinA2 * cosA3);
            w = (sinA1 * sinA2 * sinA3) + (cosA1 * cosA2 * cosA3);
        }

        /// <summary>
        /// Constructs a <see cref="Quat"/> that will rotate around the given axis
        /// by the specified angle. The axis must be a normalized vector.
        /// </summary>
        /// <param name="axis">The axis to rotate around. Must be normalized.</param>
        /// <param name="angle">The angle to rotate, in radians.</param>
        public Quat(Vector3 axis, real_t angle)
        {
#if DEBUG
            if (!axis.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(axis));
            }
#endif

            real_t d = axis.Length();

            if (d == 0f)
            {
                x = 0f;
                y = 0f;
                z = 0f;
                w = 0f;
            }
            else
            {
                real_t sinAngle = Mathf.Sin(angle * 0.5f);
                real_t cosAngle = Mathf.Cos(angle * 0.5f);
                real_t s = sinAngle / d;

                x = axis.x * s;
                y = axis.y * s;
                z = axis.z * s;
                w = cosAngle;
            }
        }

        /// <summary>
        /// Composes these two quaternions by multiplying them together.
        /// This has the effect of rotating the second quaternion
        /// (the child) by the first quaternion (the parent).
        /// </summary>
        /// <param name="left">The parent quaternion.</param>
        /// <param name="right">The child quaternion.</param>
        /// <returns>The composed quaternion.</returns>
        public static Quat operator *(Quat left, Quat right)
        {
            return new Quat
            (
                (left.w * right.x) + (left.x * right.w) + (left.y * right.z) - (left.z * right.y),
                (left.w * right.y) + (left.y * right.w) + (left.z * right.x) - (left.x * right.z),
                (left.w * right.z) + (left.z * right.w) + (left.x * right.y) - (left.y * right.x),
                (left.w * right.w) - (left.x * right.x) - (left.y * right.y) - (left.z * right.z)
            );
        }

        /// <summary>
        /// Adds each component of the left <see cref="Quat"/>
        /// to the right <see cref="Quat"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression, such as approximating an intermediate
        /// rotation between two nearby rotations.
        /// </summary>
        /// <param name="left">The left quaternion to add.</param>
        /// <param name="right">The right quaternion to add.</param>
        /// <returns>The added quaternion.</returns>
        public static Quat operator +(Quat left, Quat right)
        {
            return new Quat(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
        }

        /// <summary>
        /// Subtracts each component of the left <see cref="Quat"/>
        /// by the right <see cref="Quat"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The left quaternion to subtract.</param>
        /// <param name="right">The right quaternion to subtract.</param>
        /// <returns>The subtracted quaternion.</returns>
        public static Quat operator -(Quat left, Quat right)
        {
            return new Quat(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w);
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Quat"/>.
        /// This is the same as writing
        /// <c>new Quat(-q.x, -q.y, -q.z, -q.w)</c>. This operation
        /// results in a quaternion that represents the same rotation.
        /// </summary>
        /// <param name="quat">The quaternion to negate.</param>
        /// <returns>The negated quaternion.</returns>
        public static Quat operator -(Quat quat)
        {
            return new Quat(-quat.x, -quat.y, -quat.z, -quat.w);
        }

        [Obsolete("This operator does not have the correct behavior and will be replaced in the future. Do not use this.")]
        public static Quat operator *(Quat left, Vector3 right)
        {
            return new Quat
            (
                (left.w * right.x) + (left.y * right.z) - (left.z * right.y),
                (left.w * right.y) + (left.z * right.x) - (left.x * right.z),
                (left.w * right.z) + (left.x * right.y) - (left.y * right.x),
                -(left.x * right.x) - (left.y * right.y) - (left.z * right.z)
            );
        }

        [Obsolete("This operator does not have the correct behavior and will be replaced in the future. Do not use this.")]
        public static Quat operator *(Vector3 left, Quat right)
        {
            return new Quat
            (
                (right.w * left.x) + (right.y * left.z) - (right.z * left.y),
                (right.w * left.y) + (right.z * left.x) - (right.x * left.z),
                (right.w * left.z) + (right.x * left.y) - (right.y * left.x),
                -(right.x * left.x) - (right.y * left.y) - (right.z * left.z)
            );
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The quaternion to multiply.</param>
        /// <param name="right">The value to multiply by.</param>
        /// <returns>The multiplied quaternion.</returns>
        public static Quat operator *(Quat left, real_t right)
        {
            return new Quat(left.x * right, left.y * right, left.z * right, left.w * right);
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The value to multiply by.</param>
        /// <param name="right">The quaternion to multiply.</param>
        /// <returns>The multiplied quaternion.</returns>
        public static Quat operator *(real_t left, Quat right)
        {
            return new Quat(right.x * left, right.y * left, right.z * left, right.w * left);
        }

        /// <summary>
        /// Divides each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The quaternion to divide.</param>
        /// <param name="right">The value to divide by.</param>
        /// <returns>The divided quaternion.</returns>
        public static Quat operator /(Quat left, real_t right)
        {
            return left * (1.0f / right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the quaternions are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left quaternion.</param>
        /// <param name="right">The right quaternion.</param>
        /// <returns>Whether or not the quaternions are exactly equal.</returns>
        public static bool operator ==(Quat left, Quat right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the quaternions are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left quaternion.</param>
        /// <param name="right">The right quaternion.</param>
        /// <returns>Whether or not the quaternions are not equal.</returns>
        public static bool operator !=(Quat left, Quat right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this quaternion and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the quaternion and the other object are exactly equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Quat)
            {
                return Equals((Quat)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this quaternion and <paramref name="other"/> are equal.
        /// </summary>
        /// <param name="other">The other quaternion to compare.</param>
        /// <returns>Whether or not the quaternions are exactly equal.</returns>
        public bool Equals(Quat other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this quaternion and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Mathf.IsEqualApprox(real_t, real_t)"/> on each component.
        /// </summary>
        /// <param name="other">The other quaternion to compare.</param>
        /// <returns>Whether or not the quaternions are approximately equal.</returns>
        public bool IsEqualApprox(Quat other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z) && Mathf.IsEqualApprox(w, other.w);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Quat"/>.
        /// </summary>
        /// <returns>A hash code for this quaternion.</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Quat"/> to a string.
        /// </summary>
        /// <returns>A string representation of this quaternion.</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z}, {w})";
        }

        /// <summary>
        /// Converts this <see cref="Quat"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this quaternion.</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)}, {w.ToString(format)})";
        }
    }
}
