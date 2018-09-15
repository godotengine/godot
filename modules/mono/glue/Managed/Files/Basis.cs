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
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, 1f, 0f),
            new Vector3(0f, 0f, 1f)
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

        public Vector3 x
        {
            get { return GetAxis(0); }
            set { SetAxis(0, value); }
        }

        public Vector3 y
        {
            get { return GetAxis(1); }
            set { SetAxis(1, value); }
        }

        public Vector3 z
        {
            get { return GetAxis(2); }
            set { SetAxis(2, value); }
        }

        private Vector3 _x;
        private Vector3 _y;
        private Vector3 _z;

        public static Basis Identity
        {
            get { return identity; }
        }

        public Vector3 Scale
        {
            get
            {
                return new Vector3
                (
                    new Vector3(this[0, 0], this[1, 0], this[2, 0]).Length(),
                    new Vector3(this[0, 1], this[1, 1], this[2, 1]).Length(),
                    new Vector3(this[0, 2], this[1, 2], this[2, 2]).Length()
                );
            }
        }

        public Vector3 this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return _x;
                    case 1:
                        return _y;
                    case 2:
                        return _z;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        _x = value;
                        return;
                    case 1:
                        _y = value;
                        return;
                    case 2:
                        _z = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public real_t this[int index, int axis]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return _x[axis];
                    case 1:
                        return _y[axis];
                    case 2:
                        return _z[axis];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        _x[axis] = value;
                        return;
                    case 1:
                        _y[axis] = value;
                        return;
                    case 2:
                        _z[axis] = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        internal static Basis CreateFromAxes(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            return new Basis
            (
                new Vector3(xAxis.x, yAxis.x, zAxis.x),
                new Vector3(xAxis.y, yAxis.y, zAxis.y),
                new Vector3(xAxis.z, yAxis.z, zAxis.z)
            );
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
            _x = new Vector3(diagonal.x, 0, 0);
            _y = new Vector3(0, diagonal.y, 0);
            _z = new Vector3(0, 0, diagonal.z);

        }

        public real_t Determinant()
        {
            return this[0, 0] * (this[1, 1] * this[2, 2] - this[2, 1] * this[1, 2]) -
                    this[1, 0] * (this[0, 1] * this[2, 2] - this[2, 1] * this[0, 2]) +
                    this[2, 0] * (this[0, 1] * this[1, 2] - this[1, 1] * this[0, 2]);
        }

        public Vector3 GetAxis(int axis)
        {
            return new Vector3(this[0, axis], this[1, axis], this[2, axis]);
        }

        public void SetAxis(int axis, Vector3 value)
        {
            this[0, axis] = value.x;
            this[1, axis] = value.y;
            this[2, axis] = value.z;
        }

        public Vector3 GetEuler()
        {
            Basis m = Orthonormalized();

            Vector3 euler;
            euler.z = 0.0f;

            real_t mxy = m[1, 2];


            if (mxy < 1.0f)
            {
                if (mxy > -1.0f)
                {
                    euler.x = Mathf.Asin(-mxy);
                    euler.y = Mathf.Atan2(m[0, 2], m[2, 2]);
                    euler.z = Mathf.Atan2(m[1, 0], m[1, 1]);
                }
                else
                {
                    euler.x = Mathf.Pi * 0.5f;
                    euler.y = -Mathf.Atan2(-m[0, 1], m[0, 0]);
                }
            }
            else
            {
                euler.x = -Mathf.Pi * 0.5f;
                euler.y = -Mathf.Atan2(-m[0, 1], m[0, 0]);
            }

            return euler;
        }

        public int GetOrthogonalIndex()
        {
            var orth = this;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    real_t v = orth[i, j];

                    if (v > 0.5f)
                        v = 1.0f;
                    else if (v < -0.5f)
                        v = -1.0f;
                    else
                        v = 0f;

                    orth[i, j] = v;
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
            var inv = this;

            real_t[] co = {
                inv[1, 1] * inv[2, 2] - inv[1, 2] * inv[2, 1],
                inv[1, 2] * inv[2, 0] - inv[1, 0] * inv[2, 2],
                inv[1, 0] * inv[2, 1] - inv[1, 1] * inv[2, 0]
            };

            real_t det = inv[0, 0] * co[0] + inv[0, 1] * co[1] + inv[0, 2] * co[2];

            if (det == 0)
            {
                return new Basis
                (
                    real_t.NaN, real_t.NaN, real_t.NaN,
                    real_t.NaN, real_t.NaN, real_t.NaN,
                    real_t.NaN, real_t.NaN, real_t.NaN
                );
            }

            real_t s = 1.0f / det;

            inv = new Basis
            (
                co[0] * s,
                inv[0, 2] * inv[2, 1] - inv[0, 1] * inv[2, 2] * s,
                inv[0, 1] * inv[1, 2] - inv[0, 2] * inv[1, 1] * s,
                co[1] * s,
                inv[0, 0] * inv[2, 2] - inv[0, 2] * inv[2, 0] * s,
                inv[0, 2] * inv[1, 0] - inv[0, 0] * inv[1, 2] * s,
                co[2] * s,
                inv[0, 1] * inv[2, 0] - inv[0, 0] * inv[2, 1] * s,
                inv[0, 0] * inv[1, 1] - inv[0, 1] * inv[1, 0] * s
            );

            return inv;
        }

        public Basis Orthonormalized()
        {
            Vector3 xAxis = GetAxis(0);
            Vector3 yAxis = GetAxis(1);
            Vector3 zAxis = GetAxis(2);

            xAxis.Normalize();
            yAxis = yAxis - xAxis * xAxis.Dot(yAxis);
            yAxis.Normalize();
            zAxis = zAxis - xAxis * xAxis.Dot(zAxis) - yAxis * yAxis.Dot(zAxis);
            zAxis.Normalize();

            return CreateFromAxes(xAxis, yAxis, zAxis);
        }

        public Basis Rotated(Vector3 axis, real_t phi)
        {
            return new Basis(axis, phi) * this;
        }

        public Basis Scaled(Vector3 scale)
        {
            var m = this;

            m[0, 0] *= scale.x;
            m[0, 1] *= scale.x;
            m[0, 2] *= scale.x;
            m[1, 0] *= scale.y;
            m[1, 1] *= scale.y;
            m[1, 2] *= scale.y;
            m[2, 0] *= scale.z;
            m[2, 1] *= scale.z;
            m[2, 2] *= scale.z;

            return m;
        }

        public real_t Tdotx(Vector3 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1] + this[2, 0] * with[2];
        }

        public real_t Tdoty(Vector3 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1] + this[2, 1] * with[2];
        }

        public real_t Tdotz(Vector3 with)
        {
            return this[0, 2] * with[0] + this[1, 2] * with[1] + this[2, 2] * with[2];
        }

        public Basis Transposed()
        {
            var tr = this;

            real_t temp = tr[0, 1];
            tr[0, 1] = tr[1, 0];
            tr[1, 0] = temp;

            temp = tr[0, 2];
            tr[0, 2] = tr[2, 0];
            tr[2, 0] = temp;

            temp = tr[1, 2];
            tr[1, 2] = tr[2, 1];
            tr[2, 1] = temp;

            return tr;
        }

        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                this[0].Dot(v),
                this[1].Dot(v),
                this[2].Dot(v)
            );
        }

        public Vector3 XformInv(Vector3 v)
        {
            return new Vector3
            (
                this[0, 0] * v.x + this[1, 0] * v.y + this[2, 0] * v.z,
                this[0, 1] * v.x + this[1, 1] * v.y + this[2, 1] * v.z,
                this[0, 2] * v.x + this[1, 2] * v.y + this[2, 2] * v.z
            );
        }

        public Quat Quat() {
            real_t trace = _x[0] + _y[1] + _z[2];

            if (trace > 0.0f) {
                real_t s = Mathf.Sqrt(trace + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (_z[1] - _y[2]) * inv_s,
                    (_x[2] - _z[0]) * inv_s,
                    (_y[0] - _x[1]) * inv_s,
                    s * 0.25f
                );
            }

            if (_x[0] > _y[1] && _x[0] > _z[2]) {
                real_t s = Mathf.Sqrt(_x[0] - _y[1] - _z[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    s * 0.25f,
                    (_x[1] + _y[0]) * inv_s,
                    (_x[2] + _z[0]) * inv_s,
                    (_z[1] - _y[2]) * inv_s
                );
            }

            if (_y[1] > _z[2]) {
                real_t s = Mathf.Sqrt(-_x[0] + _y[1] - _z[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (_x[1] + _y[0]) * inv_s,
                    s * 0.25f,
                    (_y[2] + _z[1]) * inv_s,
                    (_x[2] - _z[0]) * inv_s
                );
            } else {
                real_t s = Mathf.Sqrt(-_x[0] - _y[1] + _z[2] + 1.0f) * 2f;
                real_t inv_s = 1f / s;
                return new Quat(
                    (_x[2] + _z[0]) * inv_s,
                    (_y[2] + _z[1]) * inv_s,
                    s * 0.25f,
                    (_y[0] - _x[1]) * inv_s
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

            _x = new Vector3(1.0f - (yy + zz), xy - wz, xz + wy);
            _y = new Vector3(xy + wz, 1.0f - (xx + zz), yz - wx);
            _z = new Vector3(xz - wy, yz + wx, 1.0f - (xx + yy));
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
            var axis_sq = new Vector3(axis.x * axis.x, axis.y * axis.y, axis.z * axis.z);

            real_t cosine = Mathf.Cos( phi);
            real_t sine = Mathf.Sin( phi);

            _x = new Vector3
            (
                axis_sq.x + cosine * (1.0f - axis_sq.x),
                axis.x * axis.y * (1.0f - cosine) - axis.z * sine,
                axis.z * axis.x * (1.0f - cosine) + axis.y * sine
            );

            _y = new Vector3
            (
                axis.x * axis.y * (1.0f - cosine) + axis.z * sine,
                axis_sq.y + cosine * (1.0f - axis_sq.y),
                axis.y * axis.z * (1.0f - cosine) - axis.x * sine
            );

            _z = new Vector3
            (
                axis.z * axis.x * (1.0f - cosine) - axis.y * sine,
                axis.y * axis.z * (1.0f - cosine) + axis.x * sine,
                axis_sq.z + cosine * (1.0f - axis_sq.z)
            );
        }

        public Basis(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            _x = xAxis;
            _y = yAxis;
            _z = zAxis;
        }

        public Basis(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz)
        {
            _x = new Vector3(xx, xy, xz);
            _y = new Vector3(yx, yy, yz);
            _z = new Vector3(zx, zy, zz);
        }

        public static Basis operator *(Basis left, Basis right)
        {
            return new Basis
            (
                right.Tdotx(left[0]), right.Tdoty(left[0]), right.Tdotz(left[0]),
                right.Tdotx(left[1]), right.Tdoty(left[1]), right.Tdotz(left[1]),
                right.Tdotx(left[2]), right.Tdoty(left[2]), right.Tdotz(left[2])
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
            return _x.Equals(other[0]) && _y.Equals(other[1]) && _z.Equals(other[2]);
        }

        public override int GetHashCode()
        {
            return _x.GetHashCode() ^ _y.GetHashCode() ^ _z.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                _x.ToString(),
                _y.ToString(),
                _z.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                _x.ToString(format),
                _y.ToString(format),
                _z.ToString(format)
            });
        }
    }
}
