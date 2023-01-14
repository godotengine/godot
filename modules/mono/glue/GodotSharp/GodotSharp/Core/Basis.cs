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
    /// https://docs.godotengine.org/en/latest/tutorials/math/matrices_and_transforms.html
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
            readonly get => Column0;
            set => Column0 = value;
        }

        /// <summary>
        /// The basis matrix's Y vector (column 1).
        /// </summary>
        /// <value>Equivalent to <see cref="Column1"/> and array index <c>[1]</c>.</value>
        public Vector3 y
        {
            readonly get => Column1;
            set => Column1 = value;
        }

        /// <summary>
        /// The basis matrix's Z vector (column 2).
        /// </summary>
        /// <value>Equivalent to <see cref="Column2"/> and array index <c>[2]</c>.</value>
        public Vector3 z
        {
            readonly get => Column2;
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
            readonly get => new Vector3(Row0.x, Row1.x, Row2.x);
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
            readonly get => new Vector3(Row0.y, Row1.y, Row2.y);
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
            readonly get => new Vector3(Row0.z, Row1.z, Row2.z);
            set
            {
                Row0.z = value.x;
                Row1.z = value.y;
                Row2.z = value.z;
            }
        }

        /// <summary>
        /// Access whole columns in the form of <see cref="Vector3"/>.
        /// </summary>
        /// <param name="column">Which column vector.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// <paramref name="column"/> is not 0, 1, 2 or 3.
        /// </exception>
        /// <value>The basis column.</value>
        public Vector3 this[int column]
        {
            readonly get
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
                        throw new ArgumentOutOfRangeException(nameof(column));
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
                        throw new ArgumentOutOfRangeException(nameof(column));
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
            readonly get
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

        internal void SetQuaternionScale(Quaternion quaternion, Vector3 scale)
        {
            SetDiagonal(scale);
            Rotate(quaternion);
        }

        private void Rotate(Quaternion quaternion)
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
        public readonly real_t Determinant()
        {
            real_t cofac00 = Row1[1] * Row2[2] - Row1[2] * Row2[1];
            real_t cofac10 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
            real_t cofac20 = Row1[0] * Row2[1] - Row1[1] * Row2[0];

            return Row0[0] * cofac00 + Row0[1] * cofac10 + Row0[2] * cofac20;
        }

        /// <summary>
        /// Returns the basis's rotation in the form of Euler angles.
        /// The Euler order depends on the [param order] parameter,
        /// by default it uses the YXZ convention: when decomposing,
        /// first Z, then X, and Y last. The returned vector contains
        /// the rotation angles in the format (X angle, Y angle, Z angle).
        ///
        /// Consider using the <see cref="GetRotationQuaternion"/> method instead, which
        /// returns a <see cref="Quaternion"/> quaternion instead of Euler angles.
        /// </summary>
        /// <param name="order">The Euler order to use. By default, use YXZ order (most common).</param>
        /// <returns>A <see cref="Vector3"/> representing the basis rotation in Euler angles.</returns>
        public readonly Vector3 GetEuler(EulerOrder order = EulerOrder.Yxz)
        {
            switch (order)
            {
                case EulerOrder.Xyz:
                {
                    // Euler angles in XYZ convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cy*cz          -cy*sz           sy
                    //        cz*sx*sy+cx*sz  cx*cz-sx*sy*sz -cy*sx
                    //       -cx*cz*sy+sx*sz  cz*sx+cx*sy*sz  cx*cy
                    Vector3 euler;
                    real_t sy = Row0[2];
                    if (sy < (1.0f - Mathf.Epsilon))
                    {
                        if (sy > -(1.0f - Mathf.Epsilon))
                        {
                            // is this a pure Y rotation?
                            if (Row1[0] == 0 && Row0[1] == 0 && Row1[2] == 0 && Row2[1] == 0 && Row1[1] == 1)
                            {
                                // return the simplest form (human friendlier in editor and scripts)
                                euler.x = 0;
                                euler.y = Mathf.Atan2(Row0[2], Row0[0]);
                                euler.z = 0;
                            }
                            else
                            {
                                euler.x = Mathf.Atan2(-Row1[2], Row2[2]);
                                euler.y = Mathf.Asin(sy);
                                euler.z = Mathf.Atan2(-Row0[1], Row0[0]);
                            }
                        }
                        else
                        {
                            euler.x = Mathf.Atan2(Row2[1], Row1[1]);
                            euler.y = -Mathf.Tau / 4.0f;
                            euler.z = 0.0f;
                        }
                    }
                    else
                    {
                        euler.x = Mathf.Atan2(Row2[1], Row1[1]);
                        euler.y = Mathf.Tau / 4.0f;
                        euler.z = 0.0f;
                    }
                    return euler;
                }
                case EulerOrder.Xzy:
                {
                    // Euler angles in XZY convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cz*cy             -sz             cz*sy
                    //        sx*sy+cx*cy*sz    cx*cz           cx*sz*sy-cy*sx
                    //        cy*sx*sz          cz*sx           cx*cy+sx*sz*sy
                    Vector3 euler;
                    real_t sz = Row0[1];
                    if (sz < (1.0f - Mathf.Epsilon))
                    {
                        if (sz > -(1.0f - Mathf.Epsilon))
                        {
                            euler.x = Mathf.Atan2(Row2[1], Row1[1]);
                            euler.y = Mathf.Atan2(Row0[2], Row0[0]);
                            euler.z = Mathf.Asin(-sz);
                        }
                        else
                        {
                            // It's -1
                            euler.x = -Mathf.Atan2(Row1[2], Row2[2]);
                            euler.y = 0.0f;
                            euler.z = Mathf.Tau / 4.0f;
                        }
                    }
                    else
                    {
                        // It's 1
                        euler.x = -Mathf.Atan2(Row1[2], Row2[2]);
                        euler.y = 0.0f;
                        euler.z = -Mathf.Tau / 4.0f;
                    }
                    return euler;
                }
                case EulerOrder.Yxz:
                {
                    // Euler angles in YXZ convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cy*cz+sy*sx*sz    cz*sy*sx-cy*sz        cx*sy
                    //        cx*sz             cx*cz                 -sx
                    //        cy*sx*sz-cz*sy    cy*cz*sx+sy*sz        cy*cx
                    Vector3 euler;
                    real_t m12 = Row1[2];
                    if (m12 < (1 - Mathf.Epsilon))
                    {
                        if (m12 > -(1 - Mathf.Epsilon))
                        {
                            // is this a pure X rotation?
                            if (Row1[0] == 0 && Row0[1] == 0 && Row0[2] == 0 && Row2[0] == 0 && Row0[0] == 1)
                            {
                                // return the simplest form (human friendlier in editor and scripts)
                                euler.x = Mathf.Atan2(-m12, Row1[1]);
                                euler.y = 0;
                                euler.z = 0;
                            }
                            else
                            {
                                euler.x = Mathf.Asin(-m12);
                                euler.y = Mathf.Atan2(Row0[2], Row2[2]);
                                euler.z = Mathf.Atan2(Row1[0], Row1[1]);
                            }
                        }
                        else
                        { // m12 == -1
                            euler.x = Mathf.Tau / 4.0f;
                            euler.y = Mathf.Atan2(Row0[1], Row0[0]);
                            euler.z = 0;
                        }
                    }
                    else
                    { // m12 == 1
                        euler.x = -Mathf.Tau / 4.0f;
                        euler.y = -Mathf.Atan2(Row0[1], Row0[0]);
                        euler.z = 0;
                    }

                    return euler;
                }
                case EulerOrder.Yzx:
                {
                    // Euler angles in YZX convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cy*cz             sy*sx-cy*cx*sz     cx*sy+cy*sz*sx
                    //        sz                cz*cx              -cz*sx
                    //        -cz*sy            cy*sx+cx*sy*sz     cy*cx-sy*sz*sx
                    Vector3 euler;
                    real_t sz = Row1[0];
                    if (sz < (1.0f - Mathf.Epsilon))
                    {
                        if (sz > -(1.0f - Mathf.Epsilon))
                        {
                            euler.x = Mathf.Atan2(-Row1[2], Row1[1]);
                            euler.y = Mathf.Atan2(-Row2[0], Row0[0]);
                            euler.z = Mathf.Asin(sz);
                        }
                        else
                        {
                            // It's -1
                            euler.x = Mathf.Atan2(Row2[1], Row2[2]);
                            euler.y = 0.0f;
                            euler.z = -Mathf.Tau / 4.0f;
                        }
                    }
                    else
                    {
                        // It's 1
                        euler.x = Mathf.Atan2(Row2[1], Row2[2]);
                        euler.y = 0.0f;
                        euler.z = Mathf.Tau / 4.0f;
                    }
                    return euler;
                }
                case EulerOrder.Zxy:
                {
                    // Euler angles in ZXY convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cz*cy-sz*sx*sy    -cx*sz                cz*sy+cy*sz*sx
                    //        cy*sz+cz*sx*sy    cz*cx                 sz*sy-cz*cy*sx
                    //        -cx*sy            sx                    cx*cy
                    Vector3 euler;
                    real_t sx = Row2[1];
                    if (sx < (1.0f - Mathf.Epsilon))
                    {
                        if (sx > -(1.0f - Mathf.Epsilon))
                        {
                            euler.x = Mathf.Asin(sx);
                            euler.y = Mathf.Atan2(-Row2[0], Row2[2]);
                            euler.z = Mathf.Atan2(-Row0[1], Row1[1]);
                        }
                        else
                        {
                            // It's -1
                            euler.x = -Mathf.Tau / 4.0f;
                            euler.y = Mathf.Atan2(Row0[2], Row0[0]);
                            euler.z = 0;
                        }
                    }
                    else
                    {
                        // It's 1
                        euler.x = Mathf.Tau / 4.0f;
                        euler.y = Mathf.Atan2(Row0[2], Row0[0]);
                        euler.z = 0;
                    }
                    return euler;
                }
                case EulerOrder.Zyx:
                {
                    // Euler angles in ZYX convention.
                    // See https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    //
                    // rot =  cz*cy             cz*sy*sx-cx*sz        sz*sx+cz*cx*cy
                    //        cy*sz             cz*cx+sz*sy*sx        cx*sz*sy-cz*sx
                    //        -sy               cy*sx                 cy*cx
                    Vector3 euler;
                    real_t sy = Row2[0];
                    if (sy < (1.0f - Mathf.Epsilon))
                    {
                        if (sy > -(1.0f - Mathf.Epsilon))
                        {
                            euler.x = Mathf.Atan2(Row2[1], Row2[2]);
                            euler.y = Mathf.Asin(-sy);
                            euler.z = Mathf.Atan2(Row1[0], Row0[0]);
                        }
                        else
                        {
                            // It's -1
                            euler.x = 0;
                            euler.y = Mathf.Tau / 4.0f;
                            euler.z = -Mathf.Atan2(Row0[1], Row1[1]);
                        }
                    }
                    else
                    {
                        // It's 1
                        euler.x = 0;
                        euler.y = -Mathf.Tau / 4.0f;
                        euler.z = -Mathf.Atan2(Row0[1], Row1[1]);
                    }
                    return euler;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(order));
            }
        }

        internal readonly Quaternion GetQuaternion()
        {
            real_t trace = Row0[0] + Row1[1] + Row2[2];

            if (trace > 0.0f)
            {
                real_t s = Mathf.Sqrt(trace + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quaternion(
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
                return new Quaternion(
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
                return new Quaternion(
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
                return new Quaternion(
                    (Row0[2] + Row2[0]) * inv_s,
                    (Row1[2] + Row2[1]) * inv_s,
                    s * 0.25f,
                    (Row1[0] - Row0[1]) * inv_s
                );
            }
        }

        /// <summary>
        /// Returns the <see cref="Basis"/>'s rotation in the form of a
        /// <see cref="Quaternion"/>. See <see cref="GetEuler"/> if you
        /// need Euler angles, but keep in mind quaternions should generally
        /// be preferred to Euler angles.
        /// </summary>
        /// <returns>The basis rotation.</returns>
        public readonly Quaternion GetRotationQuaternion()
        {
            Basis orthonormalizedBasis = Orthonormalized();
            real_t det = orthonormalizedBasis.Determinant();
            if (det < 0)
            {
                // Ensure that the determinant is 1, such that result is a proper
                // rotation matrix which can be represented by Euler angles.
                orthonormalizedBasis = orthonormalizedBasis.Scaled(-Vector3.One);
            }

            return orthonormalizedBasis.GetQuaternion();
        }

        /// <summary>
        /// Assuming that the matrix is the combination of a rotation and scaling,
        /// return the absolute value of scaling factors along each axis.
        /// </summary>
        public readonly Vector3 GetScale()
        {
            real_t detSign = Mathf.Sign(Determinant());
            return detSign * new Vector3
            (
                Column0.Length(),
                Column1.Length(),
                Column2.Length()
            );
        }

        /// <summary>
        /// Returns the inverse of the matrix.
        /// </summary>
        /// <returns>The inverse matrix.</returns>
        public readonly Basis Inverse()
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
        /// Returns <see langword="true"/> if this basis is finite, by calling
        /// <see cref="Mathf.IsFinite"/> on each component.
        /// </summary>
        /// <returns>Whether this vector is finite or not.</returns>
        public readonly bool IsFinite()
        {
            return Row0.IsFinite() && Row1.IsFinite() && Row2.IsFinite();
        }

        internal readonly Basis Lerp(Basis to, real_t weight)
        {
            Basis b = this;
            b.Row0 = Row0.Lerp(to.Row0, weight);
            b.Row1 = Row1.Lerp(to.Row1, weight);
            b.Row2 = Row2.Lerp(to.Row2, weight);
            return b;
        }

        /// <summary>
        /// Returns the orthonormalized version of the basis matrix (useful to
        /// call occasionally to avoid rounding errors for orthogonal matrices).
        /// This performs a Gram-Schmidt orthonormalization on the basis of the matrix.
        /// </summary>
        /// <returns>An orthonormalized basis matrix.</returns>
        public readonly Basis Orthonormalized()
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
        public readonly Basis Rotated(Vector3 axis, real_t angle)
        {
            return new Basis(axis, angle) * this;
        }

        /// <summary>
        /// Introduce an additional scaling specified by the given 3D scaling factor.
        /// </summary>
        /// <param name="scale">The scale to introduce.</param>
        /// <returns>The scaled basis matrix.</returns>
        public readonly Basis Scaled(Vector3 scale)
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
        public readonly Basis Slerp(Basis target, real_t weight)
        {
            Quaternion from = new Quaternion(this);
            Quaternion to = new Quaternion(target);

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
        public readonly real_t Tdotx(Vector3 with)
        {
            return Row0[0] * with[0] + Row1[0] * with[1] + Row2[0] * with[2];
        }

        /// <summary>
        /// Transposed dot product with the Y axis of the matrix.
        /// </summary>
        /// <param name="with">A vector to calculate the dot product with.</param>
        /// <returns>The resulting dot product.</returns>
        public readonly real_t Tdoty(Vector3 with)
        {
            return Row0[1] * with[0] + Row1[1] * with[1] + Row2[1] * with[2];
        }

        /// <summary>
        /// Transposed dot product with the Z axis of the matrix.
        /// </summary>
        /// <param name="with">A vector to calculate the dot product with.</param>
        /// <returns>The resulting dot product.</returns>
        public readonly real_t Tdotz(Vector3 with)
        {
            return Row0[2] * with[0] + Row1[2] * with[1] + Row2[2] * with[2];
        }

        /// <summary>
        /// Returns the transposed version of the basis matrix.
        /// </summary>
        /// <returns>The transposed basis matrix.</returns>
        public readonly Basis Transposed()
        {
            Basis tr = this;

            tr.Row0[1] = Row1[0];
            tr.Row1[0] = Row0[1];

            tr.Row0[2] = Row2[0];
            tr.Row2[0] = Row0[2];

            tr.Row1[2] = Row2[1];
            tr.Row2[1] = Row1[2];

            return tr;
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
        public Basis(Quaternion quaternion)
        {
            real_t s = 2.0f / quaternion.LengthSquared();

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
        /// Constructs a pure rotation basis matrix, rotated around the given <paramref name="axis"/>
        /// by <paramref name="angle"/> (in radians). The axis must be a normalized vector.
        /// </summary>
        /// <param name="axis">The axis to rotate around. Must be normalized.</param>
        /// <param name="angle">The angle to rotate, in radians.</param>
        public Basis(Vector3 axis, real_t angle)
        {
            Vector3 axisSq = new Vector3(axis.x * axis.x, axis.y * axis.y, axis.z * axis.z);
            real_t cosine = Mathf.Cos(angle);
            Row0.x = axisSq.x + cosine * (1.0f - axisSq.x);
            Row1.y = axisSq.y + cosine * (1.0f - axisSq.y);
            Row2.z = axisSq.z + cosine * (1.0f - axisSq.z);

            real_t sine = Mathf.Sin(angle);
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

        /// <summary>
        /// Constructs a transformation matrix from the given components.
        /// Arguments are named such that xy is equal to calling <c>x.y</c>.
        /// </summary>
        /// <param name="xx">The X component of the X column vector, accessed via <c>b.x.x</c> or <c>[0][0]</c>.</param>
        /// <param name="yx">The X component of the Y column vector, accessed via <c>b.y.x</c> or <c>[1][0]</c>.</param>
        /// <param name="zx">The X component of the Z column vector, accessed via <c>b.z.x</c> or <c>[2][0]</c>.</param>
        /// <param name="xy">The Y component of the X column vector, accessed via <c>b.x.y</c> or <c>[0][1]</c>.</param>
        /// <param name="yy">The Y component of the Y column vector, accessed via <c>b.y.y</c> or <c>[1][1]</c>.</param>
        /// <param name="zy">The Y component of the Z column vector, accessed via <c>b.y.y</c> or <c>[2][1]</c>.</param>
        /// <param name="xz">The Z component of the X column vector, accessed via <c>b.x.y</c> or <c>[0][2]</c>.</param>
        /// <param name="yz">The Z component of the Y column vector, accessed via <c>b.y.y</c> or <c>[1][2]</c>.</param>
        /// <param name="zz">The Z component of the Z column vector, accessed via <c>b.y.y</c> or <c>[2][2]</c>.</param>
        public Basis(real_t xx, real_t yx, real_t zx, real_t xy, real_t yy, real_t zy, real_t xz, real_t yz, real_t zz)
        {
            Row0 = new Vector3(xx, yx, zx);
            Row1 = new Vector3(xy, yy, zy);
            Row2 = new Vector3(xz, yz, zz);
        }

        /// <summary>
        /// Constructs a Basis matrix from Euler angles in the specified rotation order. By default, use YXZ order (most common).
        /// </summary>
        /// <param name="euler">The Euler angles to use.</param>
        /// <param name="order">The order to compose the Euler angles.</param>
        public static Basis FromEuler(Vector3 euler, EulerOrder order = EulerOrder.Yxz)
        {
            real_t c, s;

            c = Mathf.Cos(euler.x);
            s = Mathf.Sin(euler.x);
            Basis xmat = new Basis(new Vector3(1, 0, 0), new Vector3(0, c, s), new Vector3(0, -s, c));

            c = Mathf.Cos(euler.y);
            s = Mathf.Sin(euler.y);
            Basis ymat = new Basis(new Vector3(c, 0, -s), new Vector3(0, 1, 0), new Vector3(s, 0, c));

            c = Mathf.Cos(euler.z);
            s = Mathf.Sin(euler.z);
            Basis zmat = new Basis(new Vector3(c, s, 0), new Vector3(-s, c, 0), new Vector3(0, 0, 1));

            switch (order)
            {
                case EulerOrder.Xyz:
                    return xmat * ymat * zmat;
                case EulerOrder.Xzy:
                    return xmat * zmat * ymat;
                case EulerOrder.Yxz:
                    return ymat * xmat * zmat;
                case EulerOrder.Yzx:
                    return ymat * zmat * xmat;
                case EulerOrder.Zxy:
                    return zmat * xmat * ymat;
                case EulerOrder.Zyx:
                    return zmat * ymat * xmat;
                default:
                    throw new ArgumentOutOfRangeException(nameof(order));
            }
        }

        /// <summary>
        /// Constructs a pure scale basis matrix with no rotation or shearing.
        /// The scale values are set as the main diagonal of the matrix,
        /// and all of the other parts of the matrix are zero.
        /// </summary>
        /// <param name="scale">The scale Vector3.</param>
        /// <returns>A pure scale Basis matrix.</returns>
        public static Basis FromScale(Vector3 scale)
        {
            return new Basis(
                scale.x, 0, 0,
                0, scale.y, 0,
                0, 0, scale.z
            );
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
        /// Returns a Vector3 transformed (multiplied) by the basis matrix.
        /// </summary>
        /// <param name="basis">The basis matrix transformation to apply.</param>
        /// <param name="vector">A Vector3 to transform.</param>
        /// <returns>The transformed Vector3.</returns>
        public static Vector3 operator *(Basis basis, Vector3 vector)
        {
            return new Vector3
            (
                basis.Row0.Dot(vector),
                basis.Row1.Dot(vector),
                basis.Row2.Dot(vector)
            );
        }

        /// <summary>
        /// Returns a Vector3 transformed (multiplied) by the transposed basis matrix.
        ///
        /// Note: This results in a multiplication by the inverse of the
        /// basis matrix only if it represents a rotation-reflection.
        /// </summary>
        /// <param name="vector">A Vector3 to inversely transform.</param>
        /// <param name="basis">The basis matrix transformation to apply.</param>
        /// <returns>The inversely transformed vector.</returns>
        public static Vector3 operator *(Vector3 vector, Basis basis)
        {
            return new Vector3
            (
                basis.Row0[0] * vector.x + basis.Row1[0] * vector.y + basis.Row2[0] * vector.z,
                basis.Row0[1] * vector.x + basis.Row1[1] * vector.y + basis.Row2[1] * vector.z,
                basis.Row0[2] * vector.x + basis.Row1[2] * vector.y + basis.Row2[2] * vector.z
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
        public override readonly bool Equals(object obj)
        {
            return obj is Basis other && Equals(other);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the basis matrices are exactly
        /// equal. Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="other">The other basis.</param>
        /// <returns>Whether or not the basis matrices are exactly equal.</returns>
        public readonly bool Equals(Basis other)
        {
            return Row0.Equals(other.Row0) && Row1.Equals(other.Row1) && Row2.Equals(other.Row2);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this basis and <paramref name="other"/> are approximately equal,
        /// by running <see cref="Vector3.IsEqualApprox(Vector3)"/> on each component.
        /// </summary>
        /// <param name="other">The other basis to compare.</param>
        /// <returns>Whether or not the bases are approximately equal.</returns>
        public readonly bool IsEqualApprox(Basis other)
        {
            return Row0.IsEqualApprox(other.Row0) && Row1.IsEqualApprox(other.Row1) && Row2.IsEqualApprox(other.Row2);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Basis"/>.
        /// </summary>
        /// <returns>A hash code for this basis.</returns>
        public override readonly int GetHashCode()
        {
            return Row0.GetHashCode() ^ Row1.GetHashCode() ^ Row2.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Basis"/> to a string.
        /// </summary>
        /// <returns>A string representation of this basis.</returns>
        public override readonly string ToString()
        {
            return $"[X: {x}, Y: {y}, Z: {z}]";
        }

        /// <summary>
        /// Converts this <see cref="Basis"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this basis.</returns>
        public readonly string ToString(string format)
        {
            return $"[X: {x.ToString(format)}, Y: {y.ToString(format)}, Z: {z.ToString(format)}]";
        }
    }
}
