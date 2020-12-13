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
    /// 2×3 matrix (2 rows, 3 columns) used for 2D linear transformations.
    /// It can represent transformations such as translation, rotation, or scaling.
    /// It consists of a three <see cref="Vector2"/> values: x, y, and the origin.
    ///
    /// For more information, read this documentation article:
    /// https://docs.godotengine.org/en/latest/tutorials/math/matrices_and_transforms.html
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform2D : IEquatable<Transform2D>
    {
        /// <summary>
        /// The basis matrix's X vector (column 0). Equivalent to array index `[0]`.
        /// </summary>
        /// <value></value>
        public Vector2 x;

        /// <summary>
        /// The basis matrix's Y vector (column 1). Equivalent to array index `[1]`.
        /// </summary>
        public Vector2 y;

        /// <summary>
        /// The origin vector (column 2, the third column). Equivalent to array index `[2]`.
        /// The origin vector represents translation.
        /// </summary>
        public Vector2 origin;

        /// <summary>
        /// The rotation of this transformation matrix.
        /// </summary>
        /// <value>Getting is equivalent to calling <see cref="Mathf.Atan2(real_t, real_t)"/> with the values of <see cref="x"/>.</value>
        public real_t Rotation
        {
            get
            {
                return Mathf.Atan2(x.y, x.x);
            }
            set
            {
                Vector2 scale = Scale;
                x.x = y.y = Mathf.Cos(value);
                x.y = y.x = Mathf.Sin(value);
                y.x *= -1;
                Scale = scale;
            }
        }

        /// <summary>
        /// The scale of this transformation matrix.
        /// </summary>
        /// <value>Equivalent to the lengths of each column vector, but Y is negative if the determinant is negative.</value>
        public Vector2 Scale
        {
            get
            {
                real_t detSign = Mathf.Sign(BasisDeterminant());
                return new Vector2(x.Length(), detSign * y.Length());
            }
            set
            {
                value /= Scale; // Value becomes what's called "delta_scale" in core.
                x *= value.x;
                y *= value.y;
            }
        }

        /// <summary>
        /// Access whole columns in the form of Vector2. The third column is the origin vector.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        public Vector2 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return origin;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (column)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        origin = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Access matrix elements in column-major order. The third column is the origin vector.
        /// </summary>
        /// <param name="column">Which column, the matrix horizontal position.</param>
        /// <param name="row">Which row, the matrix vertical position.</param>
        public real_t this[int column, int row]
        {
            get
            {
                return this[column][row];
            }
            set
            {
                Vector2 columnVector = this[column];
                columnVector[row] = value;
                this[column] = columnVector;
            }
        }

        /// <summary>
        /// Returns the inverse of the transform, under the assumption that
        /// the transformation is composed of rotation, scaling, and translation.
        /// </summary>
        /// <returns>The inverse transformation matrix.</returns>
        public Transform2D AffineInverse()
        {
            real_t det = BasisDeterminant();

            if (det == 0)
                throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

            var inv = this;

            real_t temp = inv[0, 0];
            inv[0, 0] = inv[1, 1];
            inv[1, 1] = temp;

            real_t detInv = 1.0f / det;

            inv[0] *= new Vector2(detInv, -detInv);
            inv[1] *= new Vector2(-detInv, detInv);

            inv[2] = inv.BasisXform(-inv[2]);

            return inv;
        }

        /// <summary>
        /// Returns the determinant of the basis matrix. If the basis is
        /// uniformly scaled, its determinant is the square of the scale.
        ///
        /// A negative determinant means the Y scale is negative.
        /// A zero determinant means the basis isn't invertible,
        /// and is usually considered invalid.
        /// </summary>
        /// <returns>The determinant of the basis matrix.</returns>
        private real_t BasisDeterminant()
        {
            return x.x * y.y - x.y * y.x;
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the basis matrix.
        /// This method does not account for translation (the origin vector).
        /// </summary>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public Vector2 BasisXform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v));
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the inverse basis matrix.
        /// This method does not account for translation (the origin vector).
        ///
        /// Note: This results in a multiplication by the inverse of the
        /// basis matrix only if it represents a rotation-reflection.
        /// </summary>
        /// <param name="v">A vector to inversely transform.</param>
        /// <returns>The inversely transformed vector.</returns>
        public Vector2 BasisXformInv(Vector2 v)
        {
            return new Vector2(x.Dot(v), y.Dot(v));
        }

        /// <summary>
        /// Interpolates this transform to the other `transform` by `weight`.
        /// </summary>
        /// <param name="transform">The other transform.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The interpolated transform.</returns>
        public Transform2D InterpolateWith(Transform2D transform, real_t weight)
        {
            real_t r1 = Rotation;
            real_t r2 = transform.Rotation;

            Vector2 s1 = Scale;
            Vector2 s2 = transform.Scale;

            // Slerp rotation
            var v1 = new Vector2(Mathf.Cos(r1), Mathf.Sin(r1));
            var v2 = new Vector2(Mathf.Cos(r2), Mathf.Sin(r2));

            real_t dot = v1.Dot(v2);

            dot = Mathf.Clamp(dot, -1.0f, 1.0f);

            Vector2 v;

            if (dot > 0.9995f)
            {
                // Linearly interpolate to avoid numerical precision issues
                v = v1.Lerp(v2, weight).Normalized();
            }
            else
            {
                real_t angle = weight * Mathf.Acos(dot);
                Vector2 v3 = (v2 - v1 * dot).Normalized();
                v = v1 * Mathf.Cos(angle) + v3 * Mathf.Sin(angle);
            }

            // Extract parameters
            Vector2 p1 = origin;
            Vector2 p2 = transform.origin;

            // Construct matrix
            var res = new Transform2D(Mathf.Atan2(v.y, v.x), p1.Lerp(p2, weight));
            Vector2 scale = s1.Lerp(s2, weight);
            res.x *= scale;
            res.y *= scale;

            return res;
        }

        /// <summary>
        /// Returns the inverse of the transform, under the assumption that
        /// the transformation is composed of rotation and translation
        /// (no scaling, use <see cref="AffineInverse"/> for transforms with scaling).
        /// </summary>
        /// <returns>The inverse matrix.</returns>
        public Transform2D Inverse()
        {
            var inv = this;

            // Swap
            real_t temp = inv.x.y;
            inv.x.y = inv.y.x;
            inv.y.x = temp;

            inv.origin = inv.BasisXform(-inv.origin);

            return inv;
        }

        /// <summary>
        /// Returns the transform with the basis orthogonal (90 degrees),
        /// and normalized axis vectors (scale of 1 or -1).
        /// </summary>
        /// <returns>The orthonormalized transform.</returns>
        public Transform2D Orthonormalized()
        {
            var on = this;

            Vector2 onX = on.x;
            Vector2 onY = on.y;

            onX.Normalize();
            onY = onY - onX * onX.Dot(onY);
            onY.Normalize();

            on.x = onX;
            on.y = onY;

            return on;
        }

        /// <summary>
        /// Rotates the transform by `phi` (in radians), using matrix multiplication.
        /// </summary>
        /// <param name="phi">The angle to rotate, in radians.</param>
        /// <returns>The rotated transformation matrix.</returns>
        public Transform2D Rotated(real_t phi)
        {
            return this * new Transform2D(phi, new Vector2());
        }

        /// <summary>
        /// Scales the transform by the given scaling factor, using matrix multiplication.
        /// </summary>
        /// <param name="scale">The scale to introduce.</param>
        /// <returns>The scaled transformation matrix.</returns>
        public Transform2D Scaled(Vector2 scale)
        {
            var copy = this;
            copy.x *= scale;
            copy.y *= scale;
            copy.origin *= scale;
            return copy;
        }

        private void ScaleBasis(Vector2 scale)
        {
            x.x *= scale.x;
            x.y *= scale.y;
            y.x *= scale.x;
            y.y *= scale.y;
        }

        private real_t Tdotx(Vector2 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1];
        }

        private real_t Tdoty(Vector2 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1];
        }

        /// <summary>
        /// Translates the transform by the given `offset`,
        /// relative to the transform's basis vectors.
        ///
        /// Unlike <see cref="Rotated"/> and <see cref="Scaled"/>,
        /// this does not use matrix multiplication.
        /// </summary>
        /// <param name="offset">The offset to translate by.</param>
        /// <returns>The translated matrix.</returns>
        public Transform2D Translated(Vector2 offset)
        {
            var copy = this;
            copy.origin += copy.BasisXform(offset);
            return copy;
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by this transformation matrix.
        /// </summary>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public Vector2 Xform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v)) + origin;
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the inverse transformation matrix.
        /// </summary>
        /// <param name="v">A vector to inversely transform.</param>
        /// <returns>The inversely transformed vector.</returns>
        public Vector2 XformInv(Vector2 v)
        {
            Vector2 vInv = v - origin;
            return new Vector2(x.Dot(vInv), y.Dot(vInv));
        }

        // Constants
        private static readonly Transform2D _identity = new Transform2D(1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipX = new Transform2D(-1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipY = new Transform2D(1, 0, 0, -1, 0, 0);

        /// <summary>
        /// The identity transform, with no translation, rotation, or scaling applied.
        /// This is used as a replacement for `Transform2D()` in GDScript.
        /// Do not use `new Transform2D()` with no arguments in C#, because it sets all values to zero.
        /// </summary>
        /// <value>Equivalent to `new Transform2D(Vector2.Right, Vector2.Down, Vector2.Zero)`.</value>
        public static Transform2D Identity { get { return _identity; } }
        /// <summary>
        /// The transform that will flip something along the X axis.
        /// </summary>
        /// <value>Equivalent to `new Transform2D(Vector2.Left, Vector2.Down, Vector2.Zero)`.</value>
        public static Transform2D FlipX { get { return _flipX; } }
        /// <summary>
        /// The transform that will flip something along the Y axis.
        /// </summary>
        /// <value>Equivalent to `new Transform2D(Vector2.Right, Vector2.Up, Vector2.Zero)`.</value>
        public static Transform2D FlipY { get { return _flipY; } }

        /// <summary>
        /// Constructs a transformation matrix from 3 vectors (matrix columns).
        /// </summary>
        /// <param name="xAxis">The X vector, or column index 0.</param>
        /// <param name="yAxis">The Y vector, or column index 1.</param>
        /// <param name="originPos">The origin vector, or column index 2.</param>
        public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 originPos)
        {
            x = xAxis;
            y = yAxis;
            origin = originPos;
        }

        /// <summary>
        /// Constructs a transformation matrix from the given components.
        /// Arguments are named such that xy is equal to calling x.y
        /// </summary>
        /// <param name="xx">The X component of the X column vector, accessed via `t.x.x` or `[0][0]`</param>
        /// <param name="xy">The Y component of the X column vector, accessed via `t.x.y` or `[0][1]`</param>
        /// <param name="yx">The X component of the Y column vector, accessed via `t.y.x` or `[1][0]`</param>
        /// <param name="yy">The Y component of the Y column vector, accessed via `t.y.y` or `[1][1]`</param>
        /// <param name="ox">The X component of the origin vector, accessed via `t.origin.x` or `[2][0]`</param>
        /// <param name="oy">The Y component of the origin vector, accessed via `t.origin.y` or `[2][1]`</param>
        public Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy)
        {
            x = new Vector2(xx, xy);
            y = new Vector2(yx, yy);
            origin = new Vector2(ox, oy);
        }

        /// <summary>
        /// Constructs a transformation matrix from a rotation value and origin vector.
        /// </summary>
        /// <param name="rot">The rotation of the new transform, in radians.</param>
        /// <param name="pos">The origin vector, or column index 2.</param>
        public Transform2D(real_t rot, Vector2 pos)
        {
            x.x = y.y = Mathf.Cos(rot);
            x.y = y.x = Mathf.Sin(rot);
            y.x *= -1;
            origin = pos;
        }

        public static Transform2D operator *(Transform2D left, Transform2D right)
        {
            left.origin = left.Xform(right.origin);

            real_t x0 = left.Tdotx(right.x);
            real_t x1 = left.Tdoty(right.x);
            real_t y0 = left.Tdotx(right.y);
            real_t y1 = left.Tdoty(right.y);

            left.x.x = x0;
            left.x.y = x1;
            left.y.x = y0;
            left.y.y = y1;

            return left;
        }

        public static bool operator ==(Transform2D left, Transform2D right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Transform2D left, Transform2D right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            return obj is Transform2D transform2D && Equals(transform2D);
        }

        public bool Equals(Transform2D other)
        {
            return x.Equals(other.x) && y.Equals(other.y) && origin.Equals(other.origin);
        }

        /// <summary>
        /// Returns true if this transform and `other` are approximately equal, by running
        /// <see cref="Vector2.IsEqualApprox(Vector2)"/> on each component.
        /// </summary>
        /// <param name="other">The other transform to compare.</param>
        /// <returns>Whether or not the matrices are approximately equal.</returns>
        public bool IsEqualApprox(Transform2D other)
        {
            return x.IsEqualApprox(other.x) && y.IsEqualApprox(other.y) && origin.IsEqualApprox(other.origin);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ origin.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(),
                y.ToString(),
                origin.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(format),
                y.ToString(format),
                origin.ToString(format)
            });
        }
    }
}
