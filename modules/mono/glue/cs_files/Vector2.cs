using System;
using System.Runtime.InteropServices;

// file: core/math/math_2d.h
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/math/math_2d.cpp
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/variant_call.cpp
// commit: 5ad9be4c24e9d7dc5672fdc42cea896622fe5685

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector2 : IEquatable<Vector2>
    {
        public float x;
        public float y;

        public float this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
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
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        internal void Normalize()
        {
            float length = x * x + y * y;

            if (length != 0f)
            {
                length = Mathf.Sqrt(length);
                x /= length;
                y /= length;
            }
        }

        private float Cross(Vector2 b)
        {
            return x * b.y - y * b.x;
        }

        public Vector2 Abs()
        {
            return new Vector2(Mathf.Abs(x), Mathf.Abs(y));
        }

        public float Angle()
        {
            return Mathf.Atan2(y, x);
        }

        public float AngleTo(Vector2 to)
        {
            return Mathf.Atan2(Cross(to), Dot(to));
        }

        public float AngleToPoint(Vector2 to)
        {
            return Mathf.Atan2(x - to.x, y - to.y);
        }

        public float Aspect()
        {
            return x / y;
        }

        public Vector2 Bounce(Vector2 n)
        {
            return -Reflect(n);
        }

        public Vector2 Clamped(float length)
        {
            Vector2 v = this;
            float l = this.Length();

            if (l > 0 && length < l)
            {
                v /= l;
                v *= length;
            }

            return v;
        }

        public Vector2 CubicInterpolate(Vector2 b, Vector2 preA, Vector2 postB, float t)
        {
            Vector2 p0 = preA;
            Vector2 p1 = this;
            Vector2 p2 = b;
            Vector2 p3 = postB;

            float t2 = t * t;
            float t3 = t2 * t;

            return 0.5f * ((p1 * 2.0f) +
                                (-p0 + p2) * t +
                                (2.0f * p0 - 5.0f * p1 + 4 * p2 - p3) * t2 +
                                (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
        }

        public float DistanceSquaredTo(Vector2 to)
        {
            return (x - to.x) * (x - to.x) + (y - to.y) * (y - to.y);
        }

        public float DistanceTo(Vector2 to)
        {
            return Mathf.Sqrt((x - to.x) * (x - to.x) + (y - to.y) * (y - to.y));
        }

        public float Dot(Vector2 with)
        {
            return x * with.x + y * with.y;
        }

        public Vector2 Floor()
        {
            return new Vector2(Mathf.Floor(x), Mathf.Floor(y));
        }

        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared() - 1.0f) < Mathf.Epsilon;
        }

        public float Length()
        {
            return Mathf.Sqrt(x * x + y * y);
        }

        public float LengthSquared()
        {
            return x * x + y * y;
        }

        public Vector2 LinearInterpolate(Vector2 b, float t)
        {
            Vector2 res = this;

            res.x += (t * (b.x - x));
            res.y += (t * (b.y - y));

            return res;
        }

        public Vector2 Normalized()
        {
            Vector2 result = this;
            result.Normalize();
            return result;
        }

        public Vector2 Reflect(Vector2 n)
        {
            return 2.0f * n * Dot(n) - this;
        }

        public Vector2 Rotated(float phi)
        {
            float rads = Angle() + phi;
            return new Vector2(Mathf.Cos(rads), Mathf.Sin(rads)) * Length();
        }

        public Vector2 Slide(Vector2 n)
        {
            return this - n * Dot(n);
        }

        public Vector2 Snapped(Vector2 by)
        {
            return new Vector2(Mathf.Stepify(x, by.x), Mathf.Stepify(y, by.y));
        }

        public Vector2 Tangent()
        {
            return new Vector2(y, -x);
        }

        public Vector2(float x, float y)
        {
            this.x = x;
            this.y = y;
        }

        public static Vector2 operator +(Vector2 left, Vector2 right)
        {
            left.x += right.x;
            left.y += right.y;
            return left;
        }

        public static Vector2 operator -(Vector2 left, Vector2 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            return left;
        }

        public static Vector2 operator -(Vector2 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            return vec;
        }

        public static Vector2 operator *(Vector2 vec, float scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            return vec;
        }

        public static Vector2 operator *(float scale, Vector2 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            return vec;
        }

        public static Vector2 operator *(Vector2 left, Vector2 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            return left;
        }

        public static Vector2 operator /(Vector2 vec, float scale)
        {
            vec.x /= scale;
            vec.y /= scale;
            return vec;
        }

        public static Vector2 operator /(Vector2 left, Vector2 right)
        {
            left.x /= right.x;
            left.y /= right.y;
            return left;
        }

        public static bool operator ==(Vector2 left, Vector2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector2 left, Vector2 right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Vector2 left, Vector2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y < right.y;
            }
            else
            {
                return left.x < right.x;
            }
        }

        public static bool operator >(Vector2 left, Vector2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y > right.y;
            }
            else
            {
                return left.x > right.x;
            }
        }

        public static bool operator <=(Vector2 left, Vector2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y <= right.y;
            }
            else
            {
                return left.x <= right.x;
            }
        }

        public static bool operator >=(Vector2 left, Vector2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y >= right.y;
            }
            else
            {
                return left.x >= right.x;
            }
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector2)
            {
                return Equals((Vector2)obj);
            }

            return false;
        }

        public bool Equals(Vector2 other)
        {
            return x == other.x && y == other.y;
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                this.x.ToString(),
                this.y.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format)
            });
        }
    }
}
