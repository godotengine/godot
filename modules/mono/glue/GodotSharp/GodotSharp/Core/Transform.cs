using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform : IEquatable<Transform>
    {
        public Basis basis;
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

        public Transform AffineInverse()
        {
            Basis basisInv = basis.Inverse();
            return new Transform(basisInv, basisInv.Xform(-origin));
        }

        public Transform InterpolateWith(Transform transform, real_t c)
        {
            /* not sure if very "efficient" but good enough? */

            Vector3 sourceScale = basis.Scale;
            Quat sourceRotation = basis.RotationQuat();
            Vector3 sourceLocation = origin;

            Vector3 destinationScale = transform.basis.Scale;
            Quat destinationRotation = transform.basis.RotationQuat();
            Vector3 destinationLocation = transform.origin;

            var interpolated = new Transform();
            interpolated.basis.SetQuatScale(sourceRotation.Slerp(destinationRotation, c).Normalized(), sourceScale.Lerp(destinationScale, c));
            interpolated.origin = sourceLocation.Lerp(destinationLocation, c);

            return interpolated;
        }

        public Transform Inverse()
        {
            Basis basisTr = basis.Transposed();
            return new Transform(basisTr, basisTr.Xform(-origin));
        }

        public Transform LookingAt(Vector3 target, Vector3 up)
        {
            var t = this;
            t.SetLookAt(origin, target, up);
            return t;
        }

        public Transform Orthonormalized()
        {
            return new Transform(basis.Orthonormalized(), origin);
        }

        public Transform Rotated(Vector3 axis, real_t phi)
        {
            return new Transform(new Basis(axis, phi), new Vector3()) * this;
        }

        public Transform Scaled(Vector3 scale)
        {
            return new Transform(basis.Scaled(scale), origin * scale);
        }

        public void SetLookAt(Vector3 eye, Vector3 target, Vector3 up)
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

        public Transform Translated(Vector3 ofs)
        {
            return new Transform(basis, new Vector3
            (
                origin[0] += basis.Row0.Dot(ofs),
                origin[1] += basis.Row1.Dot(ofs),
                origin[2] += basis.Row2.Dot(ofs)
            ));
        }

        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                basis.Row0.Dot(v) + origin.x,
                basis.Row1.Dot(v) + origin.y,
                basis.Row2.Dot(v) + origin.z
            );
        }

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

        public static Transform Identity { get { return _identity; } }
        public static Transform FlipX { get { return _flipX; } }
        public static Transform FlipY { get { return _flipY; } }
        public static Transform FlipZ { get { return _flipZ; } }

        // Constructors
        public Transform(Vector3 column0, Vector3 column1, Vector3 column2, Vector3 origin)
        {
            basis = new Basis(column0, column1, column2);
            this.origin = origin;
        }

        public Transform(Quat quat, Vector3 origin)
        {
            basis = new Basis(quat);
            this.origin = origin;
        }

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
