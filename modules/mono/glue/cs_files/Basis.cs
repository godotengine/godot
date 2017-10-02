using System;
using System.Runtime.InteropServices;

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

        private static readonly Basis[] orthoBases = new Basis[24]
        {
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

        public Vector3 x;
        public Vector3 y;
        public Vector3 z;

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
                    new Vector3(this[0, 0], this[1, 0], this[2, 0]).length(),
                    new Vector3(this[0, 1], this[1, 1], this[2, 1]).length(),
                    new Vector3(this[0, 2], this[1, 2], this[2, 2]).length()
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
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return z;
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
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public float this[int index, int axis]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return x[axis];
                    case 1:
                        return y[axis];
                    case 2:
                        return z[axis];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        x[axis] = value;
                        return;
                    case 1:
                        y[axis] = value;
                        return;
                    case 2:
                        z[axis] = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        internal static Basis create_from_axes(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            return new Basis
            (
                new Vector3(xAxis.x, yAxis.x, zAxis.x),
                new Vector3(xAxis.y, yAxis.y, zAxis.y),
                new Vector3(xAxis.z, yAxis.z, zAxis.z)
            );
        }

        public float determinant()
        {
            return this[0, 0] * (this[1, 1] * this[2, 2] - this[2, 1] * this[1, 2]) -
                    this[1, 0] * (this[0, 1] * this[2, 2] - this[2, 1] * this[0, 2]) +
                    this[2, 0] * (this[0, 1] * this[1, 2] - this[1, 1] * this[0, 2]);
        }

        public Vector3 get_axis(int axis)
        {
            return new Vector3(this[0, axis], this[1, axis], this[2, axis]);
        }

        public Vector3 get_euler()
        {
            Basis m = this.orthonormalized();

            Vector3 euler;

            euler.y = Mathf.asin(m.x[2]);

            if (euler.y < Mathf.PI * 0.5f)
            {
                if (euler.y > -Mathf.PI * 0.5f)
                {
                    euler.x = Mathf.atan2(-m.y[2], m.z[2]);
                    euler.z = Mathf.atan2(-m.x[1], m.x[0]);
                }
                else
                {
                    euler.z = 0.0f;
                    euler.x = euler.z - Mathf.atan2(m.y[0], m.y[1]);
                }
            }
            else
            {
                euler.z = 0f;
                euler.x = Mathf.atan2(m.x[1], m.y[1]) - euler.z;
            }

            return euler;
        }

        public int get_orthogonal_index()
        {
            Basis orth = this;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    float v = orth[i, j];

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

        public Basis inverse()
        {
            Basis inv = this;

            float[] co = new float[3]
            {
                inv[1, 1] * inv[2, 2] - inv[1, 2] * inv[2, 1],
                inv[1, 2] * inv[2, 0] - inv[1, 0] * inv[2, 2],
                inv[1, 0] * inv[2, 1] - inv[1, 1] * inv[2, 0]
            };

            float det = inv[0, 0] * co[0] + inv[0, 1] * co[1] + inv[0, 2] * co[2];

            if (det == 0)
            {
                return new Basis
                (
                    float.NaN, float.NaN, float.NaN,
                    float.NaN, float.NaN, float.NaN,
                    float.NaN, float.NaN, float.NaN
                );
            }

            float s = 1.0f / det;

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

        public Basis orthonormalized()
        {
            Vector3 xAxis = get_axis(0);
            Vector3 yAxis = get_axis(1);
            Vector3 zAxis = get_axis(2);

            xAxis.normalize();
            yAxis = (yAxis - xAxis * (xAxis.dot(yAxis)));
            yAxis.normalize();
            zAxis = (zAxis - xAxis * (xAxis.dot(zAxis)) - yAxis * (yAxis.dot(zAxis)));
            zAxis.normalize();

            return Basis.create_from_axes(xAxis, yAxis, zAxis);
        }

        public Basis rotated(Vector3 axis, float phi)
        {
            return this * new Basis(axis, phi);
        }

        public Basis scaled(Vector3 scale)
        {
            Basis m = this;

            m[0, 0] *= scale.x;
            m[1, 0] *= scale.x;
            m[2, 0] *= scale.x;
            m[0, 1] *= scale.y;
            m[1, 1] *= scale.y;
            m[2, 1] *= scale.y;
            m[0, 2] *= scale.z;
            m[1, 2] *= scale.z;
            m[2, 2] *= scale.z;

            return m;
        }

        public float tdotx(Vector3 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1] + this[2, 0] * with[2];
        }

        public float tdoty(Vector3 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1] + this[2, 1] * with[2];
        }

        public float tdotz(Vector3 with)
        {
            return this[0, 2] * with[0] + this[1, 2] * with[1] + this[2, 2] * with[2];
        }

        public Basis transposed()
        {
            Basis tr = this;

            float temp = this[0, 1];
            this[0, 1] = this[1, 0];
            this[1, 0] = temp;

            temp = this[0, 2];
            this[0, 2] = this[2, 0];
            this[2, 0] = temp;

            temp = this[1, 2];
            this[1, 2] = this[2, 1];
            this[2, 1] = temp;

            return tr;
        }

        public Vector3 xform(Vector3 v)
        {
            return new Vector3
            (
                this[0].dot(v),
                this[1].dot(v),
                this[2].dot(v)
            );
        }

        public Vector3 xform_inv(Vector3 v)
        {
            return new Vector3
            (
                (this[0, 0] * v.x) + (this[1, 0] * v.y) + (this[2, 0] * v.z),
                (this[0, 1] * v.x) + (this[1, 1] * v.y) + (this[2, 1] * v.z),
                (this[0, 2] * v.x) + (this[1, 2] * v.y) + (this[2, 2] * v.z)
            );
        }

        public Basis(Quat quat)
        {
            float s = 2.0f / quat.length_squared();

            float xs = quat.x * s;
            float ys = quat.y * s;
            float zs = quat.z * s;
            float wx = quat.w * xs;
            float wy = quat.w * ys;
            float wz = quat.w * zs;
            float xx = quat.x * xs;
            float xy = quat.x * ys;
            float xz = quat.x * zs;
            float yy = quat.y * ys;
            float yz = quat.y * zs;
            float zz = quat.z * zs;

            this.x = new Vector3(1.0f - (yy + zz), xy - wz, xz + wy);
            this.y = new Vector3(xy + wz, 1.0f - (xx + zz), yz - wx);
            this.z = new Vector3(xz - wy, yz + wx, 1.0f - (xx + yy));
        }

        public Basis(Vector3 axis, float phi)
        {
            Vector3 axis_sq = new Vector3(axis.x * axis.x, axis.y * axis.y, axis.z * axis.z);

            float cosine = Mathf.cos(phi);
            float sine = Mathf.sin(phi);

            this.x = new Vector3
            (
                axis_sq.x + cosine * (1.0f - axis_sq.x),
                axis.x * axis.y * (1.0f - cosine) - axis.z * sine,
                axis.z * axis.x * (1.0f - cosine) + axis.y * sine
            );

            this.y = new Vector3
            (
                axis.x * axis.y * (1.0f - cosine) + axis.z * sine,
                axis_sq.y + cosine * (1.0f - axis_sq.y),
                axis.y * axis.z * (1.0f - cosine) - axis.x * sine
            );

            this.z = new Vector3
            (
                axis.z * axis.x * (1.0f - cosine) - axis.y * sine,
                axis.y * axis.z * (1.0f - cosine) + axis.x * sine,
                axis_sq.z + cosine * (1.0f - axis_sq.z)
            );
        }

        public Basis(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            this.x = xAxis;
            this.y = yAxis;
            this.z = zAxis;
        }

        public Basis(float xx, float xy, float xz, float yx, float yy, float yz, float zx, float zy, float zz)
        {
            this.x = new Vector3(xx, xy, xz);
            this.y = new Vector3(yx, yy, yz);
            this.z = new Vector3(zx, zy, zz);
        }

        public static Basis operator *(Basis left, Basis right)
        {
            return new Basis
            (
                right.tdotx(left[0]), right.tdoty(left[0]), right.tdotz(left[0]),
                right.tdotx(left[1]), right.tdoty(left[1]), right.tdotz(left[1]),
                right.tdotx(left[2]), right.tdoty(left[2]), right.tdotz(left[2])
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
            return x.Equals(other.x) && y.Equals(other.y) && z.Equals(other.z);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(),
                this.y.ToString(),
                this.z.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format),
                this.z.ToString(format)
            });
        }
    }
}
