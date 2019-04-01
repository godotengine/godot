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
    public struct Quat : IEquatable<Quat>
    {
        public real_t x;
        public real_t y;
        public real_t z;
        public real_t w;

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

        public real_t Length
        {
            get { return Mathf.Sqrt(LengthSquared); }
        }

        public real_t LengthSquared
        {
            get { return Dot(this); }
        }

        public Quat CubicSlerp(Quat b, Quat preA, Quat postB, real_t t)
        {
            real_t t2 = (1.0f - t) * t * 2f;
            Quat sp = Slerp(b, t);
            Quat sq = preA.Slerpni(postB, t);
            return sp.Slerpni(sq, t2);
        }

        public real_t Dot(Quat b)
        {
            return x * b.x + y * b.y + z * b.z + w * b.w;
        }

        public Vector3 GetEuler()
        {
            var basis = new Basis(this);
            return basis.GetEuler();
        }

        public Quat Inverse()
        {
            return new Quat(-x, -y, -z, w);
        }

        public Quat Normalized()
        {
            return this / Length;
        }

        public void Set(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public void Set(Quat q)
        {
            this = q;
        }

        public void SetAxisAngle(Vector3 axis, real_t angle)
        {
            this = new Quat(axis, angle);
        }

        public void SetEuler(Vector3 eulerYXZ)
        {
            this = new Quat(eulerYXZ);
        }

        public Quat Slerp(Quat b, real_t t)
        {
            // Calculate cosine
            real_t cosom = x * b.x + y * b.y + z * b.z + w * b.w;

            var to1 = new Quat();

            // Adjust signs if necessary
            if (cosom < 0.0)
            {
                cosom = -cosom;
                to1.x = -b.x;
                to1.y = -b.y;
                to1.z = -b.z;
                to1.w = -b.w;
            }
            else
            {
                to1.x = b.x;
                to1.y = b.y;
                to1.z = b.z;
                to1.w = b.w;
            }

            real_t sinom, scale0, scale1;

            // Calculate coefficients
            if (1.0 - cosom > Mathf.Epsilon)
            {
                // Standard case (Slerp)
                real_t omega = Mathf.Acos(cosom);
                sinom = Mathf.Sin(omega);
                scale0 = Mathf.Sin((1.0f - t) * omega) / sinom;
                scale1 = Mathf.Sin(t * omega) / sinom;
            }
            else
            {
                // Quaternions are very close so we can do a linear interpolation
                scale0 = 1.0f - t;
                scale1 = t;
            }

            // Calculate final values
            return new Quat
            (
                scale0 * x + scale1 * to1.x,
                scale0 * y + scale1 * to1.y,
                scale0 * z + scale1 * to1.z,
                scale0 * w + scale1 * to1.w
            );
        }

        public Quat Slerpni(Quat b, real_t t)
        {
            real_t dot = Dot(b);

            if (Mathf.Abs(dot) > 0.9999f)
            {
                return this;
            }

            real_t theta = Mathf.Acos(dot);
            real_t sinT = 1.0f / Mathf.Sin(theta);
            real_t newFactor = Mathf.Sin(t * theta) * sinT;
            real_t invFactor = Mathf.Sin((1.0f - t) * theta) * sinT;

            return new Quat
            (
                invFactor * x + newFactor * b.x,
                invFactor * y + newFactor * b.y,
                invFactor * z + newFactor * b.z,
                invFactor * w + newFactor * b.w
            );
        }

        public Vector3 Xform(Vector3 v)
        {
            Quat q = this * v;
            q *= Inverse();
            return new Vector3(q.x, q.y, q.z);
        }

        // Static Readonly Properties
        public static Quat Identity { get; } = new Quat(0f, 0f, 0f, 1f);

        // Constructors
        public Quat(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared - 1) <= Mathf.Epsilon;
        }

        public Quat(Quat q)
        {
            this = q;
        }

        public Quat(Basis basis)
        {
            this = basis.Quat();
        }

        public Quat(Vector3 eulerYXZ)
        {
            real_t half_a1 = eulerYXZ.y * (real_t)0.5;
            real_t half_a2 = eulerYXZ.x * (real_t)0.5;
            real_t half_a3 = eulerYXZ.z * (real_t)0.5;

            // R = Y(a1).X(a2).Z(a3) convention for Euler angles.
            // Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
            // a3 is the angle of the first rotation, following the notation in this reference.

            real_t cos_a1 = Mathf.Cos(half_a1);
            real_t sin_a1 = Mathf.Sin(half_a1);
            real_t cos_a2 = Mathf.Cos(half_a2);
            real_t sin_a2 = Mathf.Sin(half_a2);
            real_t cos_a3 = Mathf.Cos(half_a3);
            real_t sin_a3 = Mathf.Sin(half_a3);

            x = sin_a1 * cos_a2 * sin_a3 + cos_a1 * sin_a2 * cos_a3;
            y = sin_a1 * cos_a2 * cos_a3 - cos_a1 * sin_a2 * sin_a3;
            z = -sin_a1 * sin_a2 * cos_a3 + cos_a1 * cos_a2 * sin_a3;
            w = sin_a1 * sin_a2 * sin_a3 + cos_a1 * cos_a2 * cos_a3;
        }

        public Quat(Vector3 axis, real_t angle)
        {
            real_t d = axis.Length();
            real_t angle_t = angle;

            if (d == 0f)
            {
                x = 0f;
                y = 0f;
                z = 0f;
                w = 0f;
            }
            else
            {
                real_t s = Mathf.Sin(angle_t * 0.5f) / d;

                x = axis.x * s;
                y = axis.y * s;
                z = axis.z * s;
                w = Mathf.Cos(angle_t * 0.5f);
            }
        }

        public static Quat operator *(Quat left, Quat right)
        {
            return new Quat
            (
                left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
                left.w * right.y + left.y * right.w + left.z * right.x - left.x * right.z,
                left.w * right.z + left.z * right.w + left.x * right.y - left.y * right.x,
                left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
            );
        }

        public static Quat operator +(Quat left, Quat right)
        {
            return new Quat(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
        }

        public static Quat operator -(Quat left, Quat right)
        {
            return new Quat(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w);
        }

        public static Quat operator -(Quat left)
        {
            return new Quat(-left.x, -left.y, -left.z, -left.w);
        }

        public static Quat operator *(Quat left, Vector3 right)
        {
            return new Quat
            (
                left.w * right.x + left.y * right.z - left.z * right.y,
                left.w * right.y + left.z * right.x - left.x * right.z,
                left.w * right.z + left.x * right.y - left.y * right.x,
                -left.x * right.x - left.y * right.y - left.z * right.z
            );
        }

        public static Quat operator *(Vector3 left, Quat right)
        {
            return new Quat
            (
                right.w * left.x + right.y * left.z - right.z * left.y,
                right.w * left.y + right.z * left.x - right.x * left.z,
                right.w * left.z + right.x * left.y - right.y * left.x,
                -right.x * left.x - right.y * left.y - right.z * left.z
            );
        }

        public static Quat operator *(Quat left, real_t right)
        {
            return new Quat(left.x * right, left.y * right, left.z * right, left.w * right);
        }

        public static Quat operator *(real_t left, Quat right)
        {
            return new Quat(right.x * left, right.y * left, right.z * left, right.w * left);
        }

        public static Quat operator /(Quat left, real_t right)
        {
            return left * (1.0f / right);
        }

        public static bool operator ==(Quat left, Quat right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Quat left, Quat right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Quat)
            {
                return Equals((Quat)obj);
            }

            return false;
        }

        public bool Equals(Quat other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z) && Mathf.IsEqualApprox(w, other.w);
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2}, {3})", x.ToString(), y.ToString(), z.ToString(), w.ToString());
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2}, {3})", x.ToString(format), y.ToString(format), z.ToString(format), w.ToString(format));
        }
    }
}
