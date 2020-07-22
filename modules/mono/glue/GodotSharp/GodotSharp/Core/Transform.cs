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
    /// 3×4 matrix (3 rows, 4 columns) used for 3D linear transformations.
    /// It can represent transformations such as translation, rotation, or scaling.
    /// It consists of a <see cref="Basis"/> (first 3 columns) and a
    /// <see cref="Vector3"/> for the origin (last column).
    ///
    /// For more information, read this documentation article:
    /// https://docs.godotengine.org/en/latest/tutorials/math/matrices_and_transforms.html
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform : IEquatable<Transform>
    {
        /// <summary>
        /// The <see cref="Basis"/> of this transform. Contains the X, Y, and Z basis
        /// vectors (columns 0 to 2) and is responsible for rotation and scale.
        /// </summary>
        public Basis basis;

        /// <summary>
        /// The origin vector (column 3, the fourth column). Equivalent to array index `[3]`.
        /// </summary>
        public Vector3 origin;

        /// <summary>
        /// Access whole columns in the form of Vector3. The fourth column is the origin vector.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        public Vector3 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return basis.Column0;
                    case 1:
                        return basis.Column1;
                    case 2:
                        return basis.Column2;
                    case 3:
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
                        basis.Column0 = value;
                        return;
                    case 1:
                        basis.Column1 = value;
                        return;
                    case 2:
                        basis.Column2 = value;
                        return;
                    case 3:
                        origin = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Access matrix elements in column-major order. The fourth column is the origin vector.
        /// </summary>
        /// <param name="column">Which column, the matrix horizontal position.</param>
        /// <param name="row">Which row, the matrix vertical position.</param>
        public real_t this[int column, int row]
        {
            get
            {
                if (column == 3)
                {
                    return origin[row];
                }
                return basis[column, row];
            }
            set
            {
                if (column == 3)
                {
                    origin[row] = value;
                    return;
                }
                basis[column, row] = value;
            }
        }

        /// <summary>
        /// Returns the inverse of the transform, under the assumption that
        /// the transformation is composed of rotation, scaling, and translation.
        /// </summary>
        /// <returns>The inverse transformation matrix.</returns>
        public Transform AffineInverse()
        {
            Basis basisInv = basis.Inverse();
            return new Transform(basisInv, basisInv.Xform(-origin));
        }

        /// <summary>
        /// Interpolates this transform to the other `transform` by `weight`.
        /// </summary>
        /// <param name="transform">The other transform.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The interpolated transform.</returns>
        public Transform InterpolateWith(Transform transform, real_t weight)
        {
            /* not sure if very "efficient" but good enough? */

            Vector3 sourceScale = basis.Scale;
            Quat sourceRotation = basis.RotationQuat();
            Vector3 sourceLocation = origin;

            Vector3 destinationScale = transform.basis.Scale;
            Quat destinationRotation = transform.basis.RotationQuat();
            Vector3 destinationLocation = transform.origin;

            var interpolated = new Transform();
            interpolated.basis.SetQuatScale(sourceRotation.Slerp(destinationRotation, weight).Normalized(), sourceScale.Lerp(destinationScale, weight));
            interpolated.origin = sourceLocation.Lerp(destinationLocation, weight);

            return interpolated;
        }

        /// <summary>
        /// Returns the inverse of the transform, under the assumption that
        /// the transformation is composed of rotation and translation
        /// (no scaling, use <see cref="AffineInverse"/> for transforms with scaling).
        /// </summary>
        /// <returns>The inverse matrix.</returns>
        public Transform Inverse()
        {
            Basis basisTr = basis.Transposed();
            return new Transform(basisTr, basisTr.Xform(-origin));
        }

        /// <summary>
        /// Returns a copy of the transform rotated such that its
        /// -Z axis (forward) points towards the target position.
        ///
        /// The transform will first be rotated around the given up vector,
        /// and then fully aligned to the target by a further rotation around
        /// an axis perpendicular to both the target and up vectors.
        ///
        /// Operations take place in global space.
        /// </summary>
        /// <param name="target">The object to look at.</param>
        /// <param name="up">The relative up direction</param>
        /// <returns>The resulting transform.</returns>
        public Transform LookingAt(Vector3 target, Vector3 up)
        {
            var t = this;
            t.SetLookAt(origin, target, up);
            return t;
        }

        /// <summary>
        /// Returns the transform with the basis orthogonal (90 degrees),
        /// and normalized axis vectors (scale of 1 or -1).
        /// </summary>
        /// <returns>The orthonormalized transform.</returns>
        public Transform Orthonormalized()
        {
            return new Transform(basis.Orthonormalized(), origin);
        }

        /// <summary>
        /// Rotates the transform around the given `axis` by `phi` (in radians),
        /// using matrix multiplication. The axis must be a normalized vector.
        /// </summary>
        /// <param name="axis">The axis to rotate around. Must be normalized.</param>
        /// <param name="phi">The angle to rotate, in radians.</param>
        /// <returns>The rotated transformation matrix.</returns>
        public Transform Rotated(Vector3 axis, real_t phi)
        {
            return new Transform(new Basis(axis, phi), new Vector3()) * this;
        }

        /// <summary>
        /// Scales the transform by the given 3D scaling factor, using matrix multiplication.
        /// </summary>
        /// <param name="scale">The scale to introduce.</param>
        /// <returns>The scaled transformation matrix.</returns>
        public Transform Scaled(Vector3 scale)
        {
            return new Transform(basis.Scaled(scale), origin * scale);
        }

        private void SetLookAt(Vector3 eye, Vector3 target, Vector3 up)
        {
            // Make rotation matrix
            // Z vector
            Vector3 column2 = eye - target;

            column2.Normalize();

            Vector3 column1 = up;

            Vector3 column0 = column1.Cross(column2);

            // Recompute Y = Z cross X
            column1 = column2.Cross(column0);

            column0.Normalize();
            column1.Normalize();

            basis = new Basis(column0, column1, column2);

            origin = eye;
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
        public Transform Translated(Vector3 offset)
        {
            return new Transform(basis, new Vector3
            (
                origin[0] += basis.Row0.Dot(offset),
                origin[1] += basis.Row1.Dot(offset),
                origin[2] += basis.Row2.Dot(offset)
            ));
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by this transformation matrix.
        /// </summary>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                basis.Row0.Dot(v) + origin.x,
                basis.Row1.Dot(v) + origin.y,
                basis.Row2.Dot(v) + origin.z
            );
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the transposed transformation matrix.
        ///
        /// Note: This results in a multiplication by the inverse of the
        /// transformation matrix only if it represents a rotation-reflection.
        /// </summary>
        /// <param name="v">A vector to inversely transform.</param>
        /// <returns>The inversely transformed vector.</returns>
        public Vector3 XformInv(Vector3 v)
        {
            Vector3 vInv = v - origin;

            return new Vector3
            (
                basis.Row0[0] * vInv.x + basis.Row1[0] * vInv.y + basis.Row2[0] * vInv.z,
                basis.Row0[1] * vInv.x + basis.Row1[1] * vInv.y + basis.Row2[1] * vInv.z,
                basis.Row0[2] * vInv.x + basis.Row1[2] * vInv.y + basis.Row2[2] * vInv.z
            );
        }

        // Constants
        private static readonly Transform _identity = new Transform(Basis.Identity, Vector3.Zero);
        private static readonly Transform _flipX = new Transform(new Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1), Vector3.Zero);
        private static readonly Transform _flipY = new Transform(new Basis(1, 0, 0, 0, -1, 0, 0, 0, 1), Vector3.Zero);
        private static readonly Transform _flipZ = new Transform(new Basis(1, 0, 0, 0, 1, 0, 0, 0, -1), Vector3.Zero);

        /// <summary>
        /// The identity transform, with no translation, rotation, or scaling applied.
        /// This is used as a replacement for `Transform()` in GDScript.
        /// Do not use `new Transform()` with no arguments in C#, because it sets all values to zero.
        /// </summary>
        /// <value>Equivalent to `new Transform(Vector3.Right, Vector3.Up, Vector3.Back, Vector3.Zero)`.</value>
        public static Transform Identity { get { return _identity; } }
        /// <summary>
        /// The transform that will flip something along the X axis.
        /// </summary>
        /// <value>Equivalent to `new Transform(Vector3.Left, Vector3.Up, Vector3.Back, Vector3.Zero)`.</value>
        public static Transform FlipX { get { return _flipX; } }
        /// <summary>
        /// The transform that will flip something along the Y axis.
        /// </summary>
        /// <value>Equivalent to `new Transform(Vector3.Right, Vector3.Down, Vector3.Back, Vector3.Zero)`.</value>
        public static Transform FlipY { get { return _flipY; } }
        /// <summary>
        /// The transform that will flip something along the Z axis.
        /// </summary>
        /// <value>Equivalent to `new Transform(Vector3.Right, Vector3.Up, Vector3.Forward, Vector3.Zero)`.</value>
        public static Transform FlipZ { get { return _flipZ; } }

        /// <summary>
        /// Constructs a transformation matrix from 4 vectors (matrix columns).
        /// </summary>
        /// <param name="column0">The X vector, or column index 0.</param>
        /// <param name="column1">The Y vector, or column index 1.</param>
        /// <param name="column2">The Z vector, or column index 2.</param>
        /// <param name="origin">The origin vector, or column index 3.</param>
        public Transform(Vector3 column0, Vector3 column1, Vector3 column2, Vector3 origin)
        {
            basis = new Basis(column0, column1, column2);
            this.origin = origin;
        }

        /// <summary>
        /// Constructs a transformation matrix from the given quaternion and origin vector.
        /// </summary>
        /// <param name="quat">The <see cref="Godot.Quat"/> to create the basis from.</param>
        /// <param name="origin">The origin vector, or column index 3.</param>
        public Transform(Quat quat, Vector3 origin)
        {
            basis = new Basis(quat);
            this.origin = origin;
        }

        /// <summary>
        /// Constructs a transformation matrix from the given basis and origin vector.
        /// </summary>
        /// <param name="basis">The <see cref="Godot.Basis"/> to create the basis from.</param>
        /// <param name="origin">The origin vector, or column index 3.</param>
        public Transform(Basis basis, Vector3 origin)
        {
            this.basis = basis;
            this.origin = origin;
        }

        public static Transform operator *(Transform left, Transform right)
        {
            left.origin = left.Xform(right.origin);
            left.basis *= right.basis;
            return left;
        }

        public static bool operator ==(Transform left, Transform right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Transform left, Transform right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Transform)
            {
                return Equals((Transform)obj);
            }

            return false;
        }

        public bool Equals(Transform other)
        {
            return basis.Equals(other.basis) && origin.Equals(other.origin);
        }

        /// <summary>
        /// Returns true if this transform and `other` are approximately equal, by running
        /// <see cref="Vector3.IsEqualApprox(Vector3)"/> on each component.
        /// </summary>
        /// <param name="other">The other transform to compare.</param>
        /// <returns>Whether or not the matrices are approximately equal.</returns>
        public bool IsEqualApprox(Transform other)
        {
            return basis.IsEqualApprox(other.basis) && origin.IsEqualApprox(other.origin);
        }

        public override int GetHashCode()
        {
            return basis.GetHashCode() ^ origin.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("{0} - {1}", new object[]
            {
                basis.ToString(),
                origin.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("{0} - {1}", new object[]
            {
                basis.ToString(format),
                origin.ToString(format)
            });
        }
    }
}
