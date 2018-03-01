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
        private static readonly Quat identity = new Quat(0f, 0f, 0f, 1f);

        public real_t x;
        public real_t y;
        public real_t z;
        public real_t w;

        public static Quat Identity
        {
            get { return identity; }
        }

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

        public Quat Inverse()
        {
            return new Quat(-x, -y, -z, w);
        }

        public real_t Length()
        {
            return Mathf.Sqrt(LengthSquared());
        }

        public real_t LengthSquared()
        {
            return Dot(this);
        }

        public Quat Normalized()
        {
            return this / Length();
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
            this.x = q.x;
            this.y = q.y;
            this.z = q.z;
            this.w = q.w;
        }

        public Quat Slerp(Quat b, real_t t)
        {
            // Calculate cosine
            real_t cosom = x * b.x + y * b.y + z * b.z + w * b.w;

            real_t[] to1 = new real_t[4];

            // Adjust signs if necessary
            if (cosom < 0.0)
            {
                cosom = -cosom; to1[0] = -b.x;
                to1[1] = -b.y;
                to1[2] = -b.z;
                to1[3] = -b.w;
            }
            else
            {
                to1[0] = b.x;
                to1[1] = b.y;
                to1[2] = b.z;
                to1[3] = b.w;
            }

            real_t sinom, scale0, scale1;

            // Calculate coefficients
            if ((1.0 - cosom) > Mathf.Epsilon)
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
                scale0 * x + scale1 * to1[0],
                scale0 * y + scale1 * to1[1],
                scale0 * z + scale1 * to1[2],
                scale0 * w + scale1 * to1[3]
            );
        }

        public Quat Slerpni(Quat b, real_t t)
        {
            real_t dot = this.Dot(b);

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
                invFactor * this.x + newFactor * b.x,
                invFactor * this.y + newFactor * b.y,
                invFactor * this.z + newFactor * b.z,
                invFactor * this.w + newFactor * b.w
            );
        }

        public Vector3 Xform(Vector3 v)
        {
            Quat q = this * v;
            q *= this.Inverse();
            return new Vector3(q.x, q.y, q.z);
        }

        // Constructors 
        public Quat(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }   
        public Quat(Quat q)
        {                     
            this.x = q.x;
            this.y = q.y;
            this.z = q.z;
            this.w = q.w;
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
            if (obj is Vector2)
            {
                return Equals((Vector2)obj);
            }

            return false;
        }

        public bool Equals(Quat other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2}, {3})", new object[]
            {
                this.x.ToString(),
                this.y.ToString(),
                this.z.ToString(),
                this.w.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2}, {3})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format),
                this.z.ToString(format),
                this.w.ToString(format)
            });
        }
    }
}
