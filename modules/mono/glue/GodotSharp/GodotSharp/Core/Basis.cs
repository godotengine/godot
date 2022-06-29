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
    /// 3×3 matrix used for 3D rotation and scale.
    /// Almost always used as an orthogonal basis for a Transform.
    ///
    /// Contains 3 vector fields X, Y and Z as its columns, which are typically
    /// interpreted as the local basis vectors of a 3D transformation. For such use,
    /// it is composed of a scaling and a rotation matrix, in that order (M = R.S).
    ///
    /// Can also be accessed as array of 3D vectors. These vectors are normally
    /// orthogonal to each other, but are not necessarily normalized (due to scaling).
    ///
    /// For more information, read this documentation article:
    /// https://docs.godotengine.org/en/3.5/tutorials/math/matrices_and_transforms.html
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Basis : IEquatable<Basis>
    {
        // NOTE: x, y and z are public-only. Use Column0, Column1 and Column2 internally.

        /// <summary>
        /// The basis matrix's X vector (column 0).
        /// </summary>
        /// <value>Equivalent to <see cref="Column0"/> and array index <c>[0]</c>.</value>
        public Vector3 x
        {
            get => Column0;
            set => Column0 = value;
        }

        /// <summary>
        /// The basis matrix's Y vector (column 1).
        /// </summary>
        /// <value>Equivalent to <see cref="Column1"/> and array index <c>[1]</c>.</value>
        public Vector3 y
        {
            get => Column1;
            set => Column1 = value;
        }

        /// <summary>
        /// The basis matrix's Z vector (column 2).
        /// </summary>
        /// <value>Equivalent to <see cref="Column2"/> and array index <c>[2]</c>.</value>
        public Vector3 z
        {
            get => Column2;
            set => Column2 = value;
        }

        /// <summary>
        /// Row 0 of the basis matrix. Shows which vectors contribute
        /// to the X direction. Rows are not very useful for user code,
        /// but are more efficient for some internal calculations.
        /// </summary>
        public Vector3 Row0;

        /// <summary>
        /// Row 1 of the basis matrix. Shows which vectors contribute
        /// to the Y direction. Rows are not very useful for user code,
        /// but are more efficient for some internal calculations.
        /// </summary>
        public Vector3 Row1;

        /// <summary>
        /// Row 2 of the basis matrix. Shows which vectors contribute
        /// to the Z direction. Rows are not very useful for user code,
        /// but are more efficient for some internal calculations.
        /// </summary>
        public Vector3 Row2;

        /// <summary>
        /// Column 0 of the basis matrix (the X vector).
        /// </summary>
        /// <value>Equivalent to <see cref="x"/> and array index <c>[0]</c>.</value>
        public Vector3 Column0
        {
            get => new Vector3(Row0.x, Row1.x, Row2.x);
            set
            {
                Row0.x = value.x;
                Row1.x = value.y;
                Row2.x = value.z;
            }
        }

        /// <summary>
        /// Column 1 of the basis matrix (the Y vector).
        /// </summary>
        /// <value>Equivalent to <see cref="y"/> and array index <c>[1]</c>.</value>
        public Vector3 Column1
        {
            get => new Vector3(Row0.y, Row1.y, Row2.y);
            set
            {
                Row0.y = value.x;
                Row1.y = value.y;
                Row2.y = value.z;
            }
        }

        /// <summary>
        /// Column 2 of the basis matrix (the Z vector).
        /// </summary>
        /// <value>Equivalent to <see cref="z"/> and array index <c>[2]</c>.</value>
        public Vector3 Column2
        {
            get => new Vector3(Row0.z, Row1.z, Row2.z);
            set
            {
                Row0.z = value.x;
                Row1.z = value.y;
                Row2.z = value.z;
            }
        }

        /// <summary>
        /// The scale of this basis.
        /// </summary>
        /// <value>Equivalent to the lengths of each column vector, but negative if the determinant is negative.</value>
        public Vector3 Scale
        {
            get
            {
                real_t detSign = Mathf.Sign(Determinant());
                return detSign * new Vector3
                (
                    Column0.Length(),
                    Column1.Length(),
                    Column2.Length()
                );
            }
            set
            {
                value /= Scale; // Value becomes what's called "delta_scale" in core.
                Column0 *= value.x;
                Column1 *= value.y;
                Column2 *= value.z;
            }
        }

        /// <summary>
        /// Access whole columns in the form of <see cref="Vector3"/>.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        /// <value>The basis column.</value>
        public Vector3 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return Column0;
                    case 1:
                        return Column1;
                    case 2:
                        return Column2;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (column)
                {
                    case 0:
                        Column0 = value;
                        return;
                    case 1:
                        Column1 = value;
                        return;
                    case 2:
                        Column2 = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// Access matrix elements in column-major order.
        /// </summary>
        /// <param name="column">Which column, the matrix horizontal position.</param>
        /// <param name="row">Which row, the matrix vertical position.</param>
        /// <value>The matrix element.</value>
        public real_t this[int column, int row]
        {
            get
            {
                return this[column][row];
            }
            set
            {
                Vector3 columnVector = this[column];
                columnVector[row] = value;
                this[column] = columnVector;
            }
        }

        /// <summary>
        /// Returns the <see cref="Basis"/>'s rotation in the form of a
        /// <see cref="Quat"/>. See <see cref="GetEuler"/> if you
        /// need Euler angles, but keep in mind quaternions should generally
        /// be preferred to Euler angles.
        /// </summary>
        /// <returns>The basis rotation.</returns>
        public Quat RotationQuat()
        {
            Basis orthonormalizedBasis = Orthonormalized();
            real_t det = orthonormalizedBasis.Determinant();
            if (det < 0)
            {
                // Ensure that the determinant is 1, such that result is a proper
                // rotation matrix which can be represented by Euler angles.
                orthonormalizedBasis = orthonormalizedBasis.Scaled(-Vector3.One);
            }

            return orthonormalizedBasis.Quat();
        }

        internal void SetQuatScale(Quat quaternion, Vector3 scale)
        {
            SetDiagonal(scale);
            Rotate(quaternion);
        }

        private void Rotate(Quat quaternion)
        {
            this *= new Basis(quaternion);
        }

        private void SetDiagonal(Vector3 diagonal)
        {
            Row0 = new Vector3(diagonal.x, 0, 0);
            Row1 = new Vector3(0, diagonal.y, 0);
            Row2 = new Vector3(0, 0, diagonal.z);
        }

        /// <summary>
        /// Returns the determinant of the basis matrix. If the basis is
        /// uniformly scaled, its determinant is the square of the scale.
        ///
        /// A negative determinant means the basis has a negative scale.
        /// A zero determinant means the basis isn't invertible,
        /// and is usually considered invalid.
        /// </summary>
        /// <returns>The determinant of the basis matrix.</returns>
        public real_t Determinant()
        {
            real_t cofac00 = Row1[1] * Row2[2] - Row1[2] * Row2[1];
            real_t cofac10 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
            real_t cofac20 = Row1[0] * Row2[1] - Row1[1] * Row2[0];

            return Row0[0] * cofac00 + Row0[1] * cofac10 + Row0[2] * cofac20;
        }

        /// <summary>
        /// Returns the basis's rotation in the form of Euler angles
        /// (in the YXZ convention: when *decomposing*, first Z, then X, and Y last).
        /// The returned vector contains the rotation angles in
        /// the format (X angle, Y angle, Z angle).
        ///
        /// Consider using the <see cref="Quat()"/> method instead, which
        /// returns a <see cref="Godot.Quat"/> quaternion instead of Euler angles.
        /// </summary>
        /// <returns>A <see cref="Vector3"/> representing the basis rotation in Euler angles.</returns>
        public Vector3 GetEuler()
        {
            Basis m = Orthonormalized();

            Vector3 euler;
            euler.z = 0.0f;

            real_t mzy = m.Row1[2];

            if (mzy < 1.0f)
            {
                if (mzy > -1.0f)
                {
                    euler.x = Mathf.Asin(-mzy);
                    euler.y = Mathf.Atan2(m.Row0[2], m.Row2[2]);
                    euler.z = Mathf.Atan2(m.Row1[0], m.Row1[1]);
                }
                else
                {
                    euler.x = Mathf.Pi * 0.5f;
                    euler.y = -Mathf.Atan2(-m.Row0[1], m.Row0[0]);
                }
            }
            else
            {
                euler.x = -Mathf.Pi * 0.5f;
                euler.y = -Mathf.Atan2(-m.Row0[1], m.Row0[0]);
            }

            return euler;
        }

        /// <summary>
        /// Get rows by index. Rows are not very useful for user code,
        /// but are more efficient for some internal calculations.
        /// </summary>
        /// <param name="index">Which row.</param>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the <paramref name="index"/> is not 0, 1 or 2.
        /// </exception>
        /// <returns>One of <c>Row0</c>, <c>Row1</c>, or <c>Row2</c>.</returns>
        public Vector3 GetRow(int index)
        {
            switch (index)
            {
                case 0:
                    return Row0;
                case 1:
                    return Row1;
                case 2:
                    return Row2;
                default:
                    throw new IndexOutOfRangeException();
            }
        }

        /// <summary>
        /// Sets rows by index. Rows are not very useful for user code,
        /// but are more efficient for some internal calculations.
        /// </summary>
        /// <param name="index">Which row.</param>
        /// <param name="value">The vector to set the row to.</param>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the <paramref name="index"/> is not 0, 1 or 2.
        /// </exception>
        public void SetRow(int index, Vector3 value)
        {
            switch (index)
            {
                case 0:
                    Row0 = value;
                    return;
                case 1:
                    Row1 = value;
                    return;
                case 2:
                    Row2 = value;
                    return;
                default:
                    throw new IndexOutOfRangeException();
            }
        }

        /// <summary>
        /// Deprecated, please use the array operator instead.
        /// </summary>
        /// <param name="index">Which column.</param>
        /// <returns>One of `Column0`, `Column1`, or `Column2`.</returns>
        [Obsolete("GetColumn is deprecated. Use the array operator instead.")]
        public Vector3 GetColumn(int index)
        {
            return this[index];
        }

        /// <summary>
        /// Deprecated, please use the array operator instead.
        /// </summary>
        /// <param name="index">Which column.</param>
        /// <param name="value">The vector to set the column to.</param>
        [Obsolete("SetColumn is deprecated. Use the array operator instead.")]
        public void SetColumn(int index, Vector3 value)
        {
            this[index] = value;
        }

        /// <summary>
        /// Deprecated, please use the array operator instead.
        /// </summary>
        /// <param name="axis">Which column.</param>
        /// <returns>One of `Column0`, `Column1`, or `Column2`.</returns>
        [Obsolete("GetAxis is deprecated. Use the array operator instead.")]
        public Vector3 GetAxis(int axis)
        {
            return new Vector3(this.Row0[axis], this.Row1[axis], this.Row2[axis]);
        }

        /// <summary>
        /// This function considers a discretization of rotations into
        /// 24 points on unit sphere, lying along the vectors (x, y, z) with
        /// each component being either -1, 0, or 1, and returns the index
        /// of the point best representing the orientation of the object.
        /// It is mainly used by the <see cref="GridMap"/> editor.
        ///
        /// For further details, refer to the Godot source code.
        /// </summary>
        /// <returns>The orthogonal index.</returns>
        public int GetOrthogonalIndex()
        {
            var orth = this;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    var row = orth.GetRow(i);

                    real_t v = row[j];

                    if (v > 0.5f)
                    {
                        v = 1.0f;
                    }
                    else if (v < -0.5f)
                    {
                        v = -1.0f;
                    }
                    else
                    {
                        v = 0f;
                    }

                    row[j] = v;

                    orth.SetRow(i, row);
                }
            }

            for (int i = 0; i < 24; i++)
            {
                if (orth == _orthoBases[i])
                {
                    return i;
                }
            }

            return 0;
        }

        /// <summary>
        /// Returns the inverse of the matrix.
        /// </summary>
        /// <returns>The inverse matrix.</returns>
        public Basis Inverse()
        {
            real_t cofac00 = Row1[1] * Row2[2] - Row1[2] * Row2[1];
            real_t cofac10 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
            real_t cofac20 = Row1[0] * Row2[1] - Row1[1] * Row2[0];

            real_t det = Row0[0] * cofac00 + Row0[1] * cofac10 + Row0[2] * cofac20;

            if (det == 0)
            {
                throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");
            }

            real_t detInv = 1.0f / det;

            real_t cofac01 = Row0[2] * Row2[1] - Row0[1] * Row2[2];
            real_t cofac02 = Row0[1] * Row1[2] - Row0[2] * Row1[1];
            real_t cofac11 = Row0[0] * Row2[2] - Row0[2] * Row2[0];
            real_t cofac12 = Row0[2] * Row1[0] - Row0[0] * Row1[2];
            real_t cofac21 = Row0[1] * Row2[0] - Row0[0] * Row2[1];
            real_t cofac22 = Row0[0] * Row1[1] - Row0[1] * Row1[0];

            return new Basis
            (
                cofac00 * detInv, cofac01 * detInv, cofac02 * detInv,
                cofac10 * detInv, cofac11 * detInv, cofac12 * detInv,
                cofac20 * detInv, cofac21 * detInv, cofac22 * detInv
            );
        }

        /// <summary>
        /// Returns the orthonormalized version of the basis matrix (useful to
        /// call occasionally to avoid rounding errors for orthogonal matrices).
        /// This performs a Gram-Schmidt orthonormalization on the basis of the matrix.
        /// </summary>
        /// <returns>An orthonormalized basis matrix.</returns>
        public Basis Orthonormalized()
        {
            Vector3 column0 = this[0];
            Vector3 column1 = this[1];
            Vector3 column2 = this[2];

            column0.Normalize();
            column1 = column1 - column0 * column0.Dot(column1);
            column1.Normalize();
            column2 = column2 - column0 * column0.Dot(column2) - column1 * column1.Dot(column2);
            column2.Normalize();

            return new Basis(column0, column1, column2);
        }

        /// <summary>
        /// Introduce an additional rotation around the given <paramref name="axis"/>
        /// by <paramref name="angle"/> (in radians). The axis must be a normalized vector.
        /// </summary>
        /// <param name="axis">The axis to rotate around. Must be normalized.</param>
        /// <param name="angle">The angle to rotate, in radians.</param>
        /// <returns>The rotated basis matrix.</returns>
        public Basis Rotated(Vector3 axis, real_t phi)
        {
            return new Basis(axis, phi) * this;
        }

        /// <summary>
        /// Introduce an additional scaling specified by the given 3D scaling factor.
        /// </summary>
        /// <param name="scale">The scale to introduce.</param>
        /// <returns>The scaled basis matrix.</returns>
        public Basis Scaled(Vector3 scale)
        {
            Basis b = this;
            b.Row0 *= scale.x;
            b.Row1 *= scale.y;
            b.Row2 *= scale.z;
            return b;
        }

        /// <summary>
        /// Assuming that the matrix is a proper rotation matrix, slerp performs
        /// a spherical-linear interpolation with another rotation matrix.
        /// </summary>
        /// <param name="target">The destination basis for interpolation.</param>
        /// <param name="weight">A value on the range of 0.0 to 1.0, representing the amount of interpolation.</param>
        /// <returns>The resulting basis matrix of the interpolation.</returns>
        public Basis Slerp(Basis target, real_t weight)
        {
            Quat from = new Quat(this);
            Quat to = new Quat(target);

            Basis b = new Basis(from.Slerp(to, weight));
            b.Row0 *= Mathf.Lerp(Row0.Length(), target.Row0.Length(), weight);
            b.Row1 *= Mathf.Lerp(Row1.Length(), target.Row1.Length(), weight);
            b.Row2 *= Mathf.Lerp(Row2.Length(), target.Row2.Length(), weight);

            return b;
        }

        /// <summary>
        /// Transposed dot product with the X axis of the matrix.
        /// </summary>
        /// <param name="with">A vector to calculate the dot product with.</param>
        /// <returns>The resulting dot product.</returns>
        public real_t Tdotx(Vector3 with)
        {
            return Row0[0] * with[0] + Row1[0] * with[1] + Row2[0] * with[2];
        }

        /// <summary>
        /// Transposed dot product with the Y axis of the matrix.
        /// </summary>
        /// <param name="with">A vector to calculate the dot product with.</param>
        /// <returns>The resulting dot product.</returns>
        public real_t Tdoty(Vector3 with)
        {
            return Row0[1] * with[0] + Row1[1] * with[1] + Row2[1] * with[2];
        }

        /// <summary>
        /// Transposed dot product with the Z axis of the matrix.
        /// </summary>
        /// <param name="with">A vector to calculate the dot product with.</param>
        /// <returns>The resulting dot product.</returns>
        public real_t Tdotz(Vector3 with)
        {
            return Row0[2] * with[0] + Row1[2] * with[1] + Row2[2] * with[2];
        }

        /// <summary>
        /// Returns the transposed version of the basis matrix.
        /// </summary>
        /// <returns>The transposed basis matrix.</returns>
        public Basis Transposed()
        {
            Basis tr = this;

            real_t temp = tr.Row0[1];
            tr.Row0[1] = tr.Row1[0];
            tr.Row1[0] = temp;

            temp = tr.Row0[2];
            tr.Row0[2] = tr.Row2[0];
            tr.Row2[0] = temp;

            temp = tr.Row1[2];
            tr.Row1[2] = tr.Row2[1];
            tr.Row2[1] = temp;

            return tr;
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the basis matrix.
        /// </summary>
        /// <seealso cref="XformInv(Vector3)"/>
        /// <param name="v">A vector to transform.</param>
        /// <returns>The transformed vector.</returns>
        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                Row0.Dot(v),
                Row1.Dot(v),
                Row2.Dot(v)
            );
        }

        /// <summary>
        /// Returns a vector transformed (multiplied) by the transposed basis matrix.
        ///
        /// Note: This results in a multiplication by the inverse of the
        /// basis matrix only if it represents a rotation-reflection.
        /// </summary>
        /// <seealso cref="Xform(Vector3)"/>
        /// <param name="v">A vector to inversely transform.</param>
        /// <returns>The inversely transformed vector.</returns>
        public Vector3 XformInv(Vector3 v)
        {
            return new Vector3
            (
                Row0[0] * v.x + Row1[0] * v.y + Row2[0] * v.z,
                Row0[1] * v.x + Row1[1] * v.y + Row2[1] * v.z,
                Row0[2] * v.x + Row1[2] * v.y + Row2[2] * v.z
            );
        }

        /// <summary>
        /// Returns the basis's rotation in the form of a quaternion.
        /// See <see cref="GetEuler()"/> if you need Euler angles, but keep in
        /// mind that quaternions should generally be preferred to Euler angles.
        /// </summary>
        /// <returns>A <see cref="Godot.Quat"/> representing the basis's rotation.</returns>
        public Quat Quat()
        {
            real_t trace = Row0[0] + Row1[1] + Row2[2];

            if (trace > 0.0f)
            {
                real_t s = Mathf.Sqrt(trace + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (Row2[1] - Row1[2]) * inv_s,
                    (Row0[2] - Row2[0]) * inv_s,
                    (Row1[0] - Row0[1]) * inv_s,
                    s * 0.25f
                );
            }

            if (Row0[0] > Row1[1] && Row0[0] > Row2[2])
            {
                real_t s = Mathf.Sqrt(Row0[0] - Row1[1] - Row2[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    s * 0.25f,
                    (Row0[1] + Row1[0]) * inv_s,
                    (Row0[2] + Row2[0]) * inv_s,
                    (Row2[1] - Row1[2]) * inv_s
                );
            }

            if (Row1[1] > Row2[2])
            {
                real_t s = Mathf.Sqrt(-Row0[0] + Row1[1] - Row2[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (Row0[1] + Row1[0]) * inv_s,
                    s * 0.25f,
                    (Row1[2] + Row2[1]) * inv_s,
                    (Row0[2] - Row2[0]) * inv_s
                );
            }
            else
            {
                real_t s = Mathf.Sqrt(-Row0[0] - Row1[1] + Row2[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (Row0[2] + Row2[0]) * inv_s,
                    (Row1[2] + Row2[1]) * inv_s,
                    s * 0.25f,
                    (Row1[0] - Row0[1]) * inv_s
                );
            }
        }

        private static readonly Basis[] _orthoBases = {
            new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
            new Basis(0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f),
            new Basis(-1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f),
            new Basis(0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f),
            new Basis(1f, 0f, 0f, 0f, 0f, -1f, 0f, 1f, 0f),
            new Basis(0f, 0f, 1f, 1f, 0f, 0f, 0f, 1f, 0f),
            new Basis(-1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f),
            new Basis(0f, 0f, -1f, -1f, 0f, 0f, 0f, 1f, 0f),
            new Basis(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f),
            new Basis(0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f),
            new Basis(-1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f),
            new Basis(0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f),
            new Basis(1f, 0f, 0f, 0f, 0f, 1f, 0f, -1f, 0f),
            new Basis(0f, 0f, -1f, 1f, 0f, 0f, 0f, -1f, 0f),
            new Basis(-1f, 0f, 0f, 0f, 0f, -1f, 0f, -1f, 0f),
            new Basis(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f),
            new Basis(0f, 0f, 1f, 0f, 1f, 0f, -1f, 0f, 0f),
            new Basis(0f, -1f, 0f, 0f, 0f, 1f, -1f, 0f, 0f),
            new Basis(0f, 0f, -1f, 0f, -1f, 0f, -1f, 0f, 0f),
            new Basis(0f, 1f, 0f, 0f, 0f, -1f, -1f, 0f, 0f),
            new Basis(0f, 0f, 1f, 0f, -1f, 0f, 1f, 0f, 0f),
            new Basis(0f, 1f, 0f, 0f, 0f, 1f, 1f, 0f, 0f),
            new Basis(0f, 0f, -1f, 0f, 1f, 0f, 1f, 0f, 0f),
            new Basis(0f, -1f, 0f, 0f, 0f, -1f, 1f, 0f, 0f)
        };

        private static readonly Basis _identity = new Basis(1, 0, 0, 0, 1, 0, 0, 0, 1);
        private static readonly Basis _flipX = new Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1);
        private static readonly Basis _flipY = new Basis(1, 0, 0, 0, -1, 0, 0, 0, 1);
        private static readonly Basis _flipZ = new Basis(1, 0, 0, 0, 1, 0, 0, 0, -1);

        /// <summary>
        /// The identity basis, with no rotation or scaling applied.
        /// This is used as a replacement for <c>Basis()</c> in GDScript.
        /// Do not use <c>new Basis()</c> with no arguments in C#, because it sets all values to zero.
        /// </summary>
        /// <value>Equivalent to <c>new Basis(Vector3.Right, Vector3.Up, Vector3.Back)</c>.</value>
        public static Basis Identity { get { return _identity; } }
        /// <summary>
        /// The basis that will flip something along the X axis when used in a transformation.
        /// </summary>
        /// <value>Equivalent to <c>new Basis(Vector3.Left, Vector3.Up, Vector3.Back)</c>.</value>
        public static Basis FlipX { get { return _flipX; } }
        /// <summary>
        /// The basis that will flip something along the Y axis when used in a transformation.
        /// </summary>
        /// <value>Equivalent to <c>new Basis(Vector3.Right, Vector3.Down, Vector3.Back)</c>.</value>
        public static Basis FlipY { get { return _flipY; } }
        /// <summary>
        /// The basis that will flip something along the Z axis when used in a transformation.
        /// </summary>
        /// <value>Equivalent to <c>new Basis(Vector3.Right, Vector3.Up, Vector3.Forward)</c>.</value>
        public static Basis FlipZ { get { return _flipZ; } }

        /// <summary>
        /// Constructs a pure rotation basis matrix from the given quaternion.
        /// </summary>
        /// <param name="quaternion">The quaternion to create the basis from.</param>
        public Basis(Quat quaternion)
        {
            real_t s = 2.0f / quaternion.LengthSquared;

            real_t xs = quaternion.x * s;
            real_t ys = quaternion.y * s;
            real_t zs = quaternion.z * s;
            real_t wx = quaternion.w * xs;
            real_t wy = quaternion.w * ys;
            real_t wz = quaternion.w * zs;
            real_t xx = quaternion.x * xs;
            real_t xy = quaternion.x * ys;
            real_t xz = quaternion.x * zs;
            real_t yy = quaternion.y * ys;
            real_t yz = quaternion.y * zs;
            real_t zz = quaternion.z * zs;

            Row0 = new Vector3(1.0f - (yy + zz), xy - wz, xz + wy);
            Row1 = new Vector3(xy + wz, 1.0f - (xx + zz), yz - wx);
            Row2 = new Vector3(xz - wy, yz + wx, 1.0f - (xx + yy));
        }

        /// <summary>
        /// Constructs a pure rotation basis matrix from the given Euler angles
        /// (in the YXZ convention: when *composing*, first Y, then X, and Z last),
        /// given in the vector format as (X angle, Y angle, Z angle).
        ///
        /// Consider using the <see cref="Basis(Quat)"/> constructor instead, which
        /// uses a <see cref="Godot.Quat"/> quaternion instead of Euler angles.
        /// </summary>
        /// <param name="eulerYXZ">The Euler angles to create the basis from.</param>
        public Basis(Vector3 eulerYXZ)
        {
            real_t c;
            real_t s;

            c = Mathf.Cos(eulerYXZ.x);
            s = Mathf.Sin(eulerYXZ.x);
            var xmat = new Basis(1, 0, 0, 0, c, -s, 0, s, c);

            c = Mathf.Cos(eulerYXZ.y);
            s = Mathf.Sin(eulerYXZ.y);
            var ymat = new Basis(c, 0, s, 0, 1, 0, -s, 0, c);

            c = Mathf.Cos(eulerYXZ.z);
            s = Mathf.Sin(eulerYXZ.z);
            var zmat = new Basis(c, -s, 0, s, c, 0, 0, 0, 1);

            this = ymat * xmat * zmat;
        }

        /// <summary>
        /// Constructs a pure rotation basis matrix, rotated around the given <paramref name="axis"/>
        /// by <paramref name="angle"/> (in radians). The axis must be a normalized vector.
        /// </summary>
        /// <param name="axis">The axis to rotate around. Must be normalized.</param>
        /// <param name="angle">The angle to rotate, in radians.</param>
        public Basis(Vector3 axis, real_t phi)
        {
            Vector3 axisSq = new Vector3(axis.x * axis.x, axis.y * axis.y, axis.z * axis.z);
            real_t cosine = Mathf.Cos(phi);
            Row0.x = axisSq.x + cosine * (1.0f - axisSq.x);
            Row1.y = axisSq.y + cosine * (1.0f - axisSq.y);
            Row2.z = axisSq.z + cosine * (1.0f - axisSq.z);

            real_t sine = Mathf.Sin(phi);
            real_t t = 1.0f - cosine;

            real_t xyzt = axis.x * axis.y * t;
            real_t zyxs = axis.z * sine;
            Row0.y = xyzt - zyxs;
            Row1.x = xyzt + zyxs;

            xyzt = axis.x * axis.z * t;
            zyxs = axis.y * sine;
            Row0.z = xyzt + zyxs;
            Row2.x = xyzt - zyxs;

            xyzt = axis.y * axis.z * t;
            zyxs = axis.x * sine;
            Row1.z = xyzt - zyxs;
            Row2.y = xyzt + zyxs;
        }

        /// <summary>
        /// Constructs a basis matrix from 3 axis vectors (matrix columns).
        /// </summary>
        /// <param name="column0">The X vector, or Column0.</param>
        /// <param name="column1">The Y vector, or Column1.</param>
        /// <param name="column2">The Z vector, or Column2.</param>
        public Basis(Vector3 column0, Vector3 column1, Vector3 column2)
        {
            Row0 = new Vector3(column0.x, column1.x, column2.x);
            Row1 = new Vector3(column0.y, column1.y, column2.y);
            Row2 = new Vector3(column0.z, column1.z, column2.z);
            // Same as:
            // Column0 = column0;
            // Column1 = column1;
            // Column2 = column2;
            // We need to assign the struct fields here first so we can't do it that way...
        }

        // Arguments are named such that xy is equal to calling x.y
        internal Basis(real_t xx, real_t yx, real_t zx, real_t xy, real_t yy, real_t zy, real_t xz, real_t yz, real_t zz)
        {
            Row0 = new Vector3(xx, yx, zx);
            Row1 = new Vector3(xy, yy, zy);
            Row2 = new Vector3(xz, yz, zz);
        }

        /// <summary>
        /// Composes these two basis matrices by multiplying them
        /// together. This has the effect of transforming the second basis
        /// (the child) by the first basis (the parent).
        /// </summary>
        /// <param name="left">The parent basis.</param>
        /// <param name="right">The child basis.</param>
        /// <returns>The composed basis.</returns>
        public static Basis operator *(Basis left, Basis right)
        {
            return new Basis
            (
                right.Tdotx(left.Row0), right.Tdoty(left.Row0), right.Tdotz(left.Row0),
                right.Tdotx(left.Row1), right.Tdoty(left.Row1), right.Tdotz(left.Row1),
                right.Tdotx(left.Row2), right.Tdoty(left.Row2), right.Tdotz(left.Row2)
            );
        }

        /// <summary>
        /// Returns <see langword="true"/> if the basis matrices are exactly
        /// equal. Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left basis.</param>
        /// <param name="right">The right basis.</param>
        /// <returns>Whether or not the basis matrices are exactly equal.</returns>
        public static bool operator ==(Basis left, Basis right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the basis matrices are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left basis.</param>
        /// <param name="right">The right basis.</param>
        /// <returns>Whether or not the basis matrices are not equal.</returns>
        public static bool operator !=(Basis left, Basis right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Basis"/> is
        /// exactly equal to the given object (<see paramref="obj"/>).
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Whether or not the basis matrix and the object are exactly equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Basis)
            {
                return Equals((Basis)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the basis matrices are exactly
        /// equal. Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="other">The other basis.</param>
        /// <returns>Whether or not the basis matrices are exactly equal.</returns>
        public bool Equals(Basis other)
        {
            return Row0.Equals(other.Row0) && Row1.Equals(other.Row1) && Row2.Equals(other.Row2);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this basis and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Vector3.IsEqualApprox(Vector3)"/> on each component.
        /// </summary>
        /// <param name="other">The other basis to compare.</param>
        /// <returns>Whether or not the bases are approximately equal.</returns>
        public bool IsEqualApprox(Basis other)
        {
            return Row0.IsEqualApprox(other.Row0) && Row1.IsEqualApprox(other.Row1) && Row2.IsEqualApprox(other.Row2);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Basis"/>.
        /// </summary>
        /// <returns>A hash code for this basis.</returns>
        public override int GetHashCode()
        {
            return Row0.GetHashCode() ^ Row1.GetHashCode() ^ Row2.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Basis"/> to a string.
        /// </summary>
        /// <returns>A string representation of this basis.</returns>
        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                Row0.ToString(),
                Row1.ToString(),
                Row2.ToString()
            });
        }

        /// <summary>
        /// Converts this <see cref="Basis"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this basis.</returns>
        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                Row0.ToString(format),
                Row1.ToString(format),
                Row2.ToString(format)
            });
        }
    }
}
