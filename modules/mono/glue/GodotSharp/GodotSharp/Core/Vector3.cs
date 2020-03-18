// file: core/math/vector3.h
// commit: bd282ff43f23fe845f29a3e25c8efc01bd65ffb0
// file: core/math/vector3.cpp
// commit: 7ad14e7a3e6f87ddc450f7e34621eb5200808451
// file: core/variant_call.cpp
// commit: 5ad9be4c24e9d7dc5672fdc42cea896622fe5685
using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    /// <summary>
    /// 3-element structure that can be used to represent positions in 3D space or any other pair of numeric values.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3 : IEquatable<Vector3>
    {
        public enum Axis
        {
            X = 0,
            Y,
            Z
        }

        public real_t x;
        public real_t y;
        public real_t z;

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

        internal void Normalize()
        {
            real_t lengthsq = LengthSquared();

            if (lengthsq == 0)
            {
                x = y = z = 0f;
            }
            else
            {
                real_t length = Mathf.Sqrt(lengthsq);
                x /= length;
                y /= length;
                z /= length;
            }
        }

        public Vector3 Abs()
        {
            return new Vector3(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z));
        }

        public real_t AngleTo(Vector3 to)
        {
            return Mathf.Atan2(Cross(to).Length(), Dot(to));
        }

        public Vector3 Bounce(Vector3 n)
        {
            return -Reflect(n);
        }

        public Vector3 Ceil()
        {
            return new Vector3(Mathf.Ceil(x), Mathf.Ceil(y), Mathf.Ceil(z));
        }

        public Vector3 Cross(Vector3 b)
        {
            return new Vector3
            (
                y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x
            );
        }

        public Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, real_t t)
        {
            var p0 = preA;
            var p1 = this;
            var p2 = b;
            var p3 = postB;

            real_t t2 = t * t;
            real_t t3 = t2 * t;

            return 0.5f * (
                        p1 * 2.0f + (-p0 + p2) * t +
                        (2.0f * p0 - 5.0f * p1 + 4f * p2 - p3) * t2 +
                        (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
                    );
        }

        public Vector3 DirectionTo(Vector3 b)
        {
            return new Vector3(b.x - x, b.y - y, b.z - z).Normalized();
        }

        public real_t DistanceSquaredTo(Vector3 b)
        {
            return (b - this).LengthSquared();
        }

        public real_t DistanceTo(Vector3 b)
        {
            return (b - this).Length();
        }

        public real_t Dot(Vector3 b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        public Vector3 Floor()
        {
            return new Vector3(Mathf.Floor(x), Mathf.Floor(y), Mathf.Floor(z));
        }

        public Vector3 Inverse()
        {
            return new Vector3(1.0f / x, 1.0f / y, 1.0f / z);
        }

        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared() - 1.0f) < Mathf.Epsilon;
        }

        public real_t Length()
        {
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        public real_t LengthSquared()
        {
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;

            return x2 + y2 + z2;
        }

        public Vector3 LinearInterpolate(Vector3 b, real_t t)
        {
            return new Vector3
            (
                x + t * (b.x - x),
                y + t * (b.y - y),
                z + t * (b.z - z)
            );
        }

        public Vector3 MoveToward(Vector3 to, real_t delta)
        {
            var v = this;
            var vd = to - v;
            var len = vd.Length();
            return len <= delta || len < Mathf.Epsilon ? to : v + vd / len * delta;
        }

        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        public Vector3 Normalized()
        {
            var v = this;
            v.Normalize();
            return v;
        }

        public Basis Outer(Vector3 b)
        {
            return new Basis(
                x * b.x, x * b.y, x * b.z,
                y * b.x, y * b.y, y * b.z,
                z * b.x, z * b.y, z * b.z
            );
        }

        public Vector3 PosMod(real_t mod)
        {
            Vector3 v;
            v.x = Mathf.PosMod(x, mod);
            v.y = Mathf.PosMod(y, mod);
            v.z = Mathf.PosMod(z, mod);
            return v;
        }

        public Vector3 PosMod(Vector3 modv)
        {
            Vector3 v;
            v.x = Mathf.PosMod(x, modv.x);
            v.y = Mathf.PosMod(y, modv.y);
            v.z = Mathf.PosMod(z, modv.z);
            return v;
        }

        public Vector3 Project(Vector3 onNormal)
        {
            return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
        }

        public Vector3 Reflect(Vector3 n)
        {
#if DEBUG
            if (!n.IsNormalized())
                throw new ArgumentException("Argument  is not normalized", nameof(n));
#endif
            return 2.0f * n * Dot(n) - this;
        }

        public Vector3 Round()
        {
            return new Vector3(Mathf.Round(x), Mathf.Round(y), Mathf.Round(z));
        }

        public Vector3 Rotated(Vector3 axis, real_t phi)
        {
            return new Basis(axis, phi).Xform(this);
        }

        public Vector3 Sign()
        {
            Vector3 v;
            v.x = Mathf.Sign(x);
            v.y = Mathf.Sign(y);
            v.z = Mathf.Sign(z);
            return v;
        }

        public Vector3 Slerp(Vector3 b, real_t t)
        {
#if DEBUG
            if (!IsNormalized())
                throw new InvalidOperationException("Vector3 is not normalized");
#endif
            real_t theta = AngleTo(b);
            return Rotated(Cross(b), theta * t);
        }

        public Vector3 Slide(Vector3 n)
        {
            return this - n * Dot(n);
        }

        public Vector3 Snapped(Vector3 by)
        {
            return new Vector3
            (
                Mathf.Stepify(x, by.x),
                Mathf.Stepify(y, by.y),
                Mathf.Stepify(z, by.z)
            );
        }

        public Basis ToDiagonalMatrix()
        {
            return new Basis(
                x, 0f, 0f,
                0f, y, 0f,
                0f, 0f, z
            );
        }

        // Constants
        private static readonly Vector3 _zero = new Vector3(0, 0, 0);
        private static readonly Vector3 _one = new Vector3(1, 1, 1);
        private static readonly Vector3 _negOne = new Vector3(-1, -1, -1);
        private static readonly Vector3 _inf = new Vector3(Mathf.Inf, Mathf.Inf, Mathf.Inf);

        private static readonly Vector3 _up = new Vector3(0, 1, 0);
        private static readonly Vector3 _down = new Vector3(0, -1, 0);
        private static readonly Vector3 _right = new Vector3(1, 0, 0);
        private static readonly Vector3 _left = new Vector3(-1, 0, 0);
        private static readonly Vector3 _forward = new Vector3(0, 0, -1);
        private static readonly Vector3 _back = new Vector3(0, 0, 1);

        public static Vector3 Zero { get { return _zero; } }
        public static Vector3 One { get { return _one; } }
        public static Vector3 NegOne { get { return _negOne; } }
        public static Vector3 Inf { get { return _inf; } }

        public static Vector3 Up { get { return _up; } }
        public static Vector3 Down { get { return _down; } }
        public static Vector3 Right { get { return _right; } }
        public static Vector3 Left { get { return _left; } }
        public static Vector3 Forward { get { return _forward; } }
        public static Vector3 Back { get { return _back; } }

        // Constructors
        public Vector3(real_t x, real_t y, real_t z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public Vector3(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }

        public static Vector3 operator +(Vector3 left, Vector3 right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        public static Vector3 operator -(Vector3 left, Vector3 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        public static Vector3 operator -(Vector3 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            return vec;
        }

        public static Vector3 operator *(Vector3 vec, real_t scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3 operator *(real_t scale, Vector3 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        public static Vector3 operator *(Vector3 left, Vector3 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        public static Vector3 operator /(Vector3 vec, real_t divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            return vec;
        }

        public static Vector3 operator /(Vector3 vec, Vector3 divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            return vec;
        }

        public static Vector3 operator %(Vector3 vec, real_t divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            vec.z %= divisor;
            return vec;
        }

        public static Vector3 operator %(Vector3 vec, Vector3 divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            vec.z %= divisorv.z;
            return vec;
        }

        public static bool operator ==(Vector3 left, Vector3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Vector3 left, Vector3 right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Vector3 left, Vector3 right)
        {
            if (Mathf.IsEqualApprox(left.x, right.x))
            {
                if (Mathf.IsEqualApprox(left.y, right.y))
                    return left.z < right.z;
                return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >(Vector3 left, Vector3 right)
        {
            if (Mathf.IsEqualApprox(left.x, right.x))
            {
                if (Mathf.IsEqualApprox(left.y, right.y))
                    return left.z > right.z;
                return left.y > right.y;
            }

            return left.x > right.x;
        }

        public static bool operator <=(Vector3 left, Vector3 right)
        {
            if (Mathf.IsEqualApprox(left.x, right.x))
            {
                if (Mathf.IsEqualApprox(left.y, right.y))
                    return left.z <= right.z;
                return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >=(Vector3 left, Vector3 right)
        {
            if (Mathf.IsEqualApprox(left.x, right.x))
            {
                if (Mathf.IsEqualApprox(left.y, right.y))
                    return left.z >= right.z;
                return left.y > right.y;
            }

            return left.x > right.x;
        }

        public override bool Equals(object obj)
        {
            if (obj is Vector3)
            {
                return Equals((Vector3)obj);
            }

            return false;
        }

        public bool Equals(Vector3 other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        public bool IsEqualApprox(Vector3 other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z);
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(),
                y.ToString(),
                z.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(format),
                y.ToString(format),
                z.ToString(format)
            });
        }
    }
}
