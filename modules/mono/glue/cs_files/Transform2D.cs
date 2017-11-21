using System;
using System.Runtime.InteropServices;

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform2D : IEquatable<Transform2D>
    {
        private static readonly Transform2D identity = new Transform2D
        (
            new Vector2(1f, 0f),
            new Vector2(0f, 1f),
            new Vector2(0f, 0f)
        );

        public Vector2 x;
        public Vector2 y;
        public Vector2 o;

        public static Transform2D Identity
        {
            get { return identity; }
        }

        public Vector2 Origin
        {
            get { return o; }
        }

        public float Rotation
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
            Transform2D inv = this;

            float det = this[0, 0] * this[1, 1] - this[1, 0] * this[0, 1];

            if (det == 0)
            {
                return new Transform2D
                (
                    float.NaN, float.NaN,
                    float.NaN, float.NaN,
                    float.NaN, float.NaN
                );
            }

            float idet = 1.0f / det;

            float temp = this[0, 0];
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

        public Transform2D InterpolateWith(Transform2D m, float c)
        {
            float r1 = Rotation;
            float r2 = m.Rotation;

            Vector2 s1 = Scale;
            Vector2 s2 = m.Scale;

            // Slerp rotation
            Vector2 v1 = new Vector2(Mathf.Cos(r1), Mathf.Sin(r1));
            Vector2 v2 = new Vector2(Mathf.Cos(r2), Mathf.Sin(r2));

            float dot = v1.Dot(v2);

            // Clamp dot to [-1, 1]
            dot = (dot < -1.0f) ? -1.0f : ((dot > 1.0f) ? 1.0f : dot);

            Vector2 v = new Vector2();

            if (dot > 0.9995f)
            {
                // Linearly interpolate to avoid numerical precision issues
                v = v1.LinearInterpolate(v2, c).Normalized();
            }
            else
            {
                float angle = c * Mathf.Acos(dot);
                Vector2 v3 = (v2 - v1 * dot).Normalized();
                v = v1 * Mathf.Cos(angle) + v3 * Mathf.Sin(angle);
            }

            // Extract parameters
            Vector2 p1 = Origin;
            Vector2 p2 = m.Origin;

            // Construct matrix
            Transform2D res = new Transform2D(Mathf.Atan2(v.y, v.x), p1.LinearInterpolate(p2, c));
            Vector2 scale = s1.LinearInterpolate(s2, c);
            res.x *= scale;
            res.y *= scale;

            return res;
        }

        public Transform2D Inverse()
        {
            Transform2D inv = this;

            // Swap
            float temp = inv.x.y;
            inv.x.y = inv.y.x;
            inv.y.x = temp;

            inv.o = inv.BasisXform(-inv.o);

            return inv;
        }

        public Transform2D Orthonormalized()
        {
            Transform2D on = this;

            Vector2 onX = on.x;
            Vector2 onY = on.y;

            onX.Normalize();
            onY = onY - onX * (onX.Dot(onY));
            onY.Normalize();

            on.x = onX;
            on.y = onY;

            return on;
        }

        public Transform2D Rotated(float phi)
        {
            return this * new Transform2D(phi, new Vector2());
        }

        public Transform2D Scaled(Vector2 scale)
        {
            Transform2D copy = this;
            copy.x *= scale;
            copy.y *= scale;
            copy.o *= scale;
            return copy;
        }

        private float Tdotx(Vector2 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1];
        }

        private float Tdoty(Vector2 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1];
        }

        public Transform2D Translated(Vector2 offset)
        {
            Transform2D copy = this;
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

        public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 origin)
        {
            this.x = xAxis;
            this.y = yAxis;
            this.o = origin;
        }
        public Transform2D(float xx, float xy, float yx, float yy, float ox, float oy)
        {
            this.x = new Vector2(xx, xy);
            this.y = new Vector2(yx, yy);
            this.o = new Vector2(ox, oy);
        }

        public Transform2D(float rot, Vector2 pos)
        {
            float cr = Mathf.Cos(rot);
            float sr = Mathf.Sin(rot);
            x.x = cr;
            y.y = cr;
            x.y = -sr;
            y.x = sr;
            o = pos;
        }

        public static Transform2D operator *(Transform2D left, Transform2D right)
        {
            left.o = left.Xform(right.o);

            float x0, x1, y0, y1;

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
                this.x.ToString(),
                this.y.ToString(),
                this.o.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format),
                this.o.ToString(format)
            });
        }
    }
}
