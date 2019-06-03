using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Basis : IEquatable<Basis>
    {
        private static readonly Basis identity = new Basis
        (
            1f, 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f
        );

        private static readonly Basis[] orthoBases = {
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

        // NOTE: x, y and z are public-only. Use Column0, Column1 and Column2 internally.

        /// <summary>
        /// Returns the basis matrix’s x vector.
        /// This is equivalent to <see cref="Column0"/>.
        /// </summary>
        public Vector3 x
        {
            get => Column0;
            set => Column0 = value;
        }

        /// <summary>
        /// Returns the basis matrix’s y vector.
        /// This is equivalent to <see cref="Column1"/>.
        /// </summary>
        public Vector3 y
        {

            get => Column1;
            set => Column1 = value;
        }

        /// <summary>
        /// Returns the basis matrix’s z vector.
        /// This is equivalent to <see cref="Column2"/>.
        /// </summary>
        public Vector3 z
        {

            get => Column2;
            set => Column2 = value;
        }

        public Vector3 Row0;
        public Vector3 Row1;
        public Vector3 Row2;

        public Vector3 Column0
        {
            get => new Vector3(Row0.x, Row1.x, Row2.x);
            set
            {
                this.Row0.x = value.x;
                this.Row1.x = value.y;
                this.Row2.x = value.z;
            }
        }
        public Vector3 Column1
        {
            get => new Vector3(Row0.y, Row1.y, Row2.y);
            set
            {
                this.Row0.y = value.x;
                this.Row1.y = value.y;
                this.Row2.y = value.z;
            }
        }
        public Vector3 Column2
        {
            get => new Vector3(Row0.z, Row1.z, Row2.z);
            set
            {
                this.Row0.z = value.x;
                this.Row1.z = value.y;
                this.Row2.z = value.z;
            }
        }

        public static Basis Identity => identity;

        public Vector3 Scale
        {
            get
            {
                real_t detSign = Mathf.Sign(Determinant());
                return detSign * new Vector3
                (
                    new Vector3(this.Row0[0], this.Row1[0], this.Row2[0]).Length(),
                    new Vector3(this.Row0[1], this.Row1[1], this.Row2[1]).Length(),
                    new Vector3(this.Row0[2], this.Row1[2], this.Row2[2]).Length()
                );
            }
        }

        public Vector3 this[int columnIndex]
        {
            get
            {
                switch (columnIndex)
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
                switch (columnIndex)
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

        public real_t this[int columnIndex, int rowIndex]
        {
            get
            {
                switch (columnIndex)
                {
                    case 0:
                        return Column0[rowIndex];
                    case 1:
                        return Column1[rowIndex];
                    case 2:
                        return Column2[rowIndex];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (columnIndex)
                {
                    case 0:
                    {
                        var column0 = Column0;
                        column0[rowIndex] = value;
                        Column0 = column0;
                        return;
                    }
                    case 1:
                    {
                        var column1 = Column1;
                        column1[rowIndex] = value;
                        Column1 = column1;
                        return;
                    }
                    case 2:
                    {
                        var column2 = Column2;
                        column2[rowIndex] = value;
                        Column2 = column2;
                        return;
                    }
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        internal Quat RotationQuat()
        {
            Basis orthonormalizedBasis = Orthonormalized();
            real_t det = orthonormalizedBasis.Determinant();
            if (det < 0)
            {
                // Ensure that the determinant is 1, such that result is a proper rotation matrix which can be represented by Euler angles.
                orthonormalizedBasis = orthonormalizedBasis.Scaled(Vector3.NegOne);
            }

            return orthonormalizedBasis.Quat();
        }

        internal void SetQuantScale(Quat quat, Vector3 scale)
        {
            SetDiagonal(scale);
            Rotate(quat);
        }

        private void Rotate(Quat quat)
        {
            this *= new Basis(quat);
        }

        private void SetDiagonal(Vector3 diagonal)
        {
            Row0 = new Vector3(diagonal.x, 0, 0);
            Row1 = new Vector3(0, diagonal.y, 0);
            Row2 = new Vector3(0, 0, diagonal.z);

        }

        public real_t Determinant()
        {
            real_t cofac00 = Row1[1] * Row2[2] - Row1[2] * Row2[1];
            real_t cofac10 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
            real_t cofac20 = Row1[0] * Row2[1] - Row1[1] * Row2[0];

            return Row0[0] * cofac00 + Row0[1] * cofac10 + Row0[2] * cofac20;
        }

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

        public Vector3 GetColumn(int index)
        {
            return this[index];
        }

        public void SetColumn(int index, Vector3 value)
        {
            this[index] = value;
        }

        [Obsolete("GetAxis is deprecated. Use GetColumn instead.")]
        public Vector3 GetAxis(int axis)
        {
            return new Vector3(this.Row0[axis], this.Row1[axis], this.Row2[axis]);
        }

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
                        v = 1.0f;
                    else if (v < -0.5f)
                        v = -1.0f;
                    else
                        v = 0f;

                    row[j] = v;

                    orth.SetRow(i, row);
                }
            }

            for (int i = 0; i < 24; i++)
            {
                if (orthoBases[i] == orth)
                    return i;
            }

            return 0;
        }

        public Basis Inverse()
        {
            real_t cofac00 = Row1[1] * Row2[2] - Row1[2] * Row2[1];
            real_t cofac10 = Row1[2] * Row2[0] - Row1[0] * Row2[2];
            real_t cofac20 = Row1[0] * Row2[1] - Row1[1] * Row2[0];

            real_t det = Row0[0] * cofac00 + Row0[1] * cofac10 + Row0[2] * cofac20;

            if (det == 0)
                throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

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

        public Basis Orthonormalized()
        {
            Vector3 column0 = GetColumn(0);
            Vector3 column1 = GetColumn(1);
            Vector3 column2 = GetColumn(2);

            column0.Normalize();
            column1 = column1 - column0 * column0.Dot(column1);
            column1.Normalize();
            column2 = column2 - column0 * column0.Dot(column2) - column1 * column1.Dot(column2);
            column2.Normalize();

            return new Basis(column0, column1, column2);
        }

        public Basis Rotated(Vector3 axis, real_t phi)
        {
            return new Basis(axis, phi) * this;
        }

        public Basis Scaled(Vector3 scale)
        {
            var b = this;
            b.Row0 *= scale.x;
            b.Row1 *= scale.y;
            b.Row2 *= scale.z;
            return b;
        }

        public real_t Tdotx(Vector3 with)
        {
            return this.Row0[0] * with[0] + this.Row1[0] * with[1] + this.Row2[0] * with[2];
        }

        public real_t Tdoty(Vector3 with)
        {
            return this.Row0[1] * with[0] + this.Row1[1] * with[1] + this.Row2[1] * with[2];
        }

        public real_t Tdotz(Vector3 with)
        {
            return this.Row0[2] * with[0] + this.Row1[2] * with[1] + this.Row2[2] * with[2];
        }

        public Basis Transposed()
        {
            var tr = this;

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

        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                this.Row0.Dot(v),
                this.Row1.Dot(v),
                this.Row2.Dot(v)
            );
        }

        public Vector3 XformInv(Vector3 v)
        {
            return new Vector3
            (
                this.Row0[0] * v.x + this.Row1[0] * v.y + this.Row2[0] * v.z,
                this.Row0[1] * v.x + this.Row1[1] * v.y + this.Row2[1] * v.z,
                this.Row0[2] * v.x + this.Row1[2] * v.y + this.Row2[2] * v.z
            );
        }

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

        public Basis(Quat quat)
        {
            real_t s = 2.0f / quat.LengthSquared;

            real_t xs = quat.x * s;
            real_t ys = quat.y * s;
            real_t zs = quat.z * s;
            real_t wx = quat.w * xs;
            real_t wy = quat.w * ys;
            real_t wz = quat.w * zs;
            real_t xx = quat.x * xs;
            real_t xy = quat.x * ys;
            real_t xz = quat.x * zs;
            real_t yy = quat.y * ys;
            real_t yz = quat.y * zs;
            real_t zz = quat.z * zs;

            Row0 = new Vector3(1.0f - (yy + zz), xy - wz, xz + wy);
            Row1 = new Vector3(xy + wz, 1.0f - (xx + zz), yz - wx);
            Row2 = new Vector3(xz - wy, yz + wx, 1.0f - (xx + yy));
        }

        public Basis(Vector3 euler)
        {
            real_t c;
            real_t s;

            c = Mathf.Cos(euler.x);
            s = Mathf.Sin(euler.x);
            var xmat = new Basis(1, 0, 0, 0, c, -s, 0, s, c);

            c = Mathf.Cos(euler.y);
            s = Mathf.Sin(euler.y);
            var ymat = new Basis(c, 0, s, 0, 1, 0, -s, 0, c);

            c = Mathf.Cos(euler.z);
            s = Mathf.Sin(euler.z);
            var zmat = new Basis(c, -s, 0, s, c, 0, 0, 0, 1);

            this = ymat * xmat * zmat;
        }

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

        public static Basis operator *(Basis left, Basis right)
        {
            return new Basis
            (
                right.Tdotx(left.Row0), right.Tdoty(left.Row0), right.Tdotz(left.Row0),
                right.Tdotx(left.Row1), right.Tdoty(left.Row1), right.Tdotz(left.Row1),
                right.Tdotx(left.Row2), right.Tdoty(left.Row2), right.Tdotz(left.Row2)
            );
        }

        public static bool operator ==(Basis left, Basis right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Basis left, Basis right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Basis)
            {
                return Equals((Basis)obj);
            }

            return false;
        }

        public bool Equals(Basis other)
        {
            return Row0.Equals(other.Row0) && Row1.Equals(other.Row1) && Row2.Equals(other.Row2);
        }

        public override int GetHashCode()
        {
            return Row0.GetHashCode() ^ Row1.GetHashCode() ^ Row2.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                Row0.ToString(),
                Row1.ToString(),
                Row2.ToString()
            });
        }

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
