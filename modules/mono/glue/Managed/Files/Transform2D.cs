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
    public struct Transform2D : IEquatable<Transform2D>
    {
        public Vector2 x;
        public Vector2 y;
        public Vector2 origin;

        public real_t Rotation
        {
            get
            {
                real_t det = BasisDeterminant();
                Transform2D t = Orthonormalized();
                if (det < 0)
                {
                    t.ScaleBasis(new Vector2(1, -1));
                }
                return Mathf.Atan2(t.x.y, t.x.x);
            }
            set
            {
                Vector2 scale = Scale;
                x.x = y.y = Mathf.Cos(value);
                x.y = y.x = Mathf.Sin(value);
                y.x *= -1;
                Scale = scale;
            }
        }

        public Vector2 Scale
        {
            get
            {
                real_t detSign = Mathf.Sign(BasisDeterminant());
                return new Vector2(x.Length(), detSign * y.Length());
            }
            set
            {
                x = x.Normalized();
                y = y.Normalized();
                x *= value.x;
                y *= value.y;
            }
        }

        public Vector2 this[int rowIndex]
        {
            get
            {
                switch (rowIndex)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return origin;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (rowIndex)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        origin = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public real_t this[int rowIndex, int columnIndex]
        {
            get
            {
                switch (rowIndex)
                {
                    case 0:
                        return x[columnIndex];
                    case 1:
                        return y[columnIndex];
                    case 2:
                        return origin[columnIndex];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (rowIndex)
                {
                    case 0:
                        x[columnIndex] = value;
                        return;
                    case 1:
                        y[columnIndex] = value;
                        return;
                    case 2:
                        origin[columnIndex] = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public Transform2D AffineInverse()
        {
            real_t det = BasisDeterminant();

            if (det == 0)
                throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

            var inv = this;

            real_t temp = inv[0, 0];
            inv[0, 0] = inv[1, 1];
            inv[1, 1] = temp;

            real_t detInv = 1.0f / det;

            inv[0] *= new Vector2(detInv, -detInv);
            inv[1] *= new Vector2(-detInv, detInv);

            inv[2] = inv.BasisXform(-inv[2]);

            return inv;
        }

        private real_t BasisDeterminant()
        {
            return x.x * y.y - x.y * y.x;
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
            Vector2 p1 = origin;
            Vector2 p2 = m.origin;

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

            inv.origin = inv.BasisXform(-inv.origin);

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
            copy.origin *= scale;
            return copy;
        }

        private void ScaleBasis(Vector2 scale)
        {
            x.x *= scale.x;
            x.y *= scale.y;
            y.x *= scale.x;
            y.y *= scale.y;
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
            copy.origin += copy.BasisXform(offset);
            return copy;
        }

        public Vector2 Xform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v)) + origin;
        }

        public Vector2 XformInv(Vector2 v)
        {
            Vector2 vInv = v - origin;
            return new Vector2(x.Dot(vInv), y.Dot(vInv));
        }

        // Constants
        private static readonly Transform2D _identity = new Transform2D(1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipX = new Transform2D(-1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipY = new Transform2D(1, 0, 0, -1, 0, 0);

        public static Transform2D Identity => _identity;
        public static Transform2D FlipX => _flipX;
        public static Transform2D FlipY => _flipY;

        // Constructors
        public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 originPos)
        {
            x = xAxis;
            y = yAxis;
            origin = originPos;
        }

        // Arguments are named such that xy is equal to calling x.y
        public Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy)
        {
            x = new Vector2(xx, xy);
            y = new Vector2(yx, yy);
            origin = new Vector2(ox, oy);
        }

        public Transform2D(real_t rot, Vector2 pos)
        {
            x.x = y.y = Mathf.Cos(rot);
            x.y = y.x = Mathf.Sin(rot);
            y.x *= -1;
            origin = pos;
        }

        public static Transform2D operator *(Transform2D left, Transform2D right)
        {
            left.origin = left.Xform(right.origin);

            real_t x0 = left.Tdotx(right.x);
            real_t x1 = left.Tdoty(right.x);
            real_t y0 = left.Tdotx(right.y);
            real_t y1 = left.Tdoty(right.y);

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
            return obj is Transform2D transform2D && Equals(transform2D);
        }

        public bool Equals(Transform2D other)
        {
            return x.Equals(other.x) && y.Equals(other.y) && origin.Equals(other.origin);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ origin.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(),
                y.ToString(),
                origin.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(format),
                y.ToString(format),
                origin.ToString(format)
            });
        }
    }
}
