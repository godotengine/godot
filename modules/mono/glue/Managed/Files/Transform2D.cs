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
    public struct Transform2D : IEquatable<Transform2D>
    {
        public Vector2 x;
        public Vector2 y;
        public Vector2 o;

        public Vector2 Origin
        {
            get { return o; }
        }

        public real_t Rotation
        {
            get { return Mathf.Atan2(y.x, o.y); }
        }

        public Vector2 Scale
        {
            get { return new Vector2(x.Length(), y.Length()); }
        }

        public Vector2 this[int index]
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
                        return o;
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
                        o = value;
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
                        return x[axis];
                    case 1:
                        return y[axis];
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
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public Transform2D AffineInverse()
        {
            var inv = this;

            real_t det = this[0, 0] * this[1, 1] - this[1, 0] * this[0, 1];

            if (det == 0)
            {
                return new Transform2D
                (
                    float.NaN, float.NaN,
                    float.NaN, float.NaN,
                    float.NaN, float.NaN
                );
            }

            real_t idet = 1.0f / det;

            real_t temp = this[0, 0];
            this[0, 0] = this[1, 1];
            this[1, 1] = temp;

            this[0] *= new Vector2(idet, -idet);
            this[1] *= new Vector2(-idet, idet);

            this[2] = BasisXform(-this[2]);

            return inv;
        }

        public Vector2 BasisXform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v));
        }

        public Vector2 BasisXformInv(Vector2 v)
        {
            return new Vector2(x.Dot(v), y.Dot(v));
        }

        public Transform2D InterpolateWith(Transform2D m, real_t c)
        {
            real_t r1 = Rotation;
            real_t r2 = m.Rotation;

            Vector2 s1 = Scale;
            Vector2 s2 = m.Scale;

            // Slerp rotation
            var v1 = new Vector2(Mathf.Cos(r1), Mathf.Sin(r1));
            var v2 = new Vector2(Mathf.Cos(r2), Mathf.Sin(r2));

            real_t dot = v1.Dot(v2);

            // Clamp dot to [-1, 1]
            dot = dot < -1.0f ? -1.0f : (dot > 1.0f ? 1.0f : dot);

            Vector2 v;

            if (dot > 0.9995f)
            {
                // Linearly interpolate to avoid numerical precision issues
                v = v1.LinearInterpolate(v2, c).Normalized();
            }
            else
            {
                real_t angle = c * Mathf.Acos(dot);
                Vector2 v3 = (v2 - v1 * dot).Normalized();
                v = v1 * Mathf.Cos(angle) + v3 * Mathf.Sin(angle);
            }

            // Extract parameters
            Vector2 p1 = Origin;
            Vector2 p2 = m.Origin;

            // Construct matrix
            var res = new Transform2D(Mathf.Atan2(v.y, v.x), p1.LinearInterpolate(p2, c));
            Vector2 scale = s1.LinearInterpolate(s2, c);
            res.x *= scale;
            res.y *= scale;

            return res;
        }

        public Transform2D Inverse()
        {
            var inv = this;

            // Swap
            real_t temp = inv.x.y;
            inv.x.y = inv.y.x;
            inv.y.x = temp;

            inv.o = inv.BasisXform(-inv.o);

            return inv;
        }

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

        public Transform2D Rotated(real_t phi)
        {
            return this * new Transform2D(phi, new Vector2());
        }

        public Transform2D Scaled(Vector2 scale)
        {
            var copy = this;
            copy.x *= scale;
            copy.y *= scale;
            copy.o *= scale;
            return copy;
        }

        private real_t Tdotx(Vector2 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1];
        }

        private real_t Tdoty(Vector2 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1];
        }

        public Transform2D Translated(Vector2 offset)
        {
            var copy = this;
            copy.o += copy.BasisXform(offset);
            return copy;
        }

        public Vector2 Xform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v)) + o;
        }

        public Vector2 XformInv(Vector2 v)
        {
            Vector2 vInv = v - o;
            return new Vector2(x.Dot(vInv), y.Dot(vInv));
        }

        // Constants
        private static readonly Transform2D _identity = new Transform2D(new Vector2(1f, 0f), new Vector2(0f, 1f), Vector2.Zero);
        private static readonly Transform2D _flipX = new Transform2D(new Vector2(-1f, 0f), new Vector2(0f, 1f), Vector2.Zero);
        private static readonly Transform2D _flipY = new Transform2D(new Vector2(1f, 0f), new Vector2(0f, -1f), Vector2.Zero);

        public static Transform2D Identity { get { return _identity; } }
        public static Transform2D FlipX { get { return _flipX; } }
        public static Transform2D FlipY { get { return _flipY; } }
        
        // Constructors 
        public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 origin)
        {
            x = xAxis;
            y = yAxis;
            o = origin;
        }
        
        public Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy)
        {
            x = new Vector2(xx, xy);
            y = new Vector2(yx, yy);
            o = new Vector2(ox, oy);
        }

        public Transform2D(real_t rot, Vector2 pos)
        {
            real_t cr = Mathf.Cos(rot);
            real_t sr = Mathf.Sin(rot);
            x.x = cr;
            y.y = cr;
            x.y = -sr;
            y.x = sr;
            o = pos;
        }

        public static Transform2D operator *(Transform2D left, Transform2D right)
        {
            left.o = left.Xform(right.o);

            real_t x0, x1, y0, y1;

            x0 = left.Tdotx(right.x);
            x1 = left.Tdoty(right.x);
            y0 = left.Tdotx(right.y);
            y1 = left.Tdoty(right.y);

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
            if (obj is Transform2D)
            {
                return Equals((Transform2D)obj);
            }

            return false;
        }

        public bool Equals(Transform2D other)
        {
            return x.Equals(other.x) && y.Equals(other.y) && o.Equals(other.o);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ o.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(),
                y.ToString(),
                o.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(format),
                y.ToString(format),
                o.ToString(format)
            });
        }
    }
}
