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
    /// Point3 is a class similar to Vector3, except with integers.
    /// It is useful, for example, when working with points on a 3D grid.
    /// Not all concepts of Vector3 apply, such as normalization and any functions that depend on normalization. 
    /// However, new concepts can exist with Point3 such as bitwise operations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Point3 : IEquatable<Point3>
    {
        public enum Axis
        {
            X = 0,
            Y,
            Z
        }

        public int x;
        public int y;
        public int z;

        public int this[int index]
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

        public Point3 Abs()
        {
            return new Point3( Mathf.RoundToInt(Mathf.Abs(x)), Mathf.RoundToInt(Mathf.Abs(y)), Mathf.RoundToInt(Mathf.Abs(z)) );
        }

        public int DistanceSquaredTo(Point3 b)
        {
            return (b - this).LengthSquared();
        }

        public real_t DistanceTo(Point3 b)
        {
            return (b - this).Length();
        }

        public Point3 DivFloor(int div) 
        {
            Point3 p = this;
            p.x = Mathf.DivFloor(p.x, div);
            p.y = Mathf.DivFloor(p.y, div);
            p.z = Mathf.DivFloor(p.z, div);
            return p;
        }

        public Point3 DivFloor(Point3 divp) 
        {
            Point3 p = this;
            p.x = Mathf.DivFloor(p.x, divp.x);
            p.y = Mathf.DivFloor(p.y, divp.y);
            p.z = Mathf.DivFloor(p.z, divp.z);
            return p;
        }

        public int Dot(Point3 b)
        {
            return x * b.x + y * b.y + z * b.z;
        }

        public real_t Length()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        public int LengthSquared()
        {
            int x2 = x * x;
            int y2 = y * y;
            int z2 = z * z;

            return x2 + y2 + z2;
        }

        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        public Point3 Mod(int mod) 
        {
            Point3 p = this;
            p.x = Mathf.ModInt(p.x, mod);
            p.y = Mathf.ModInt(p.y, mod);
            p.z = Mathf.ModInt(p.z, mod);
            return p;
        }

        public Point3 Mod(Point3 modp) 
        {
            Point3 p = this;
            p.x = Mathf.ModInt(p.x, modp.x);
            p.y = Mathf.ModInt(p.y, modp.y);
            p.z = Mathf.ModInt(p.z, modp.z);
            return p;
        }

        public Point3 ModPow2(int mod) 
        {
            Point3 p = this;
            p.x = Mathf.ModPow2(p.x, mod);
            p.y = Mathf.ModPow2(p.y, mod);
            p.z = Mathf.ModPow2(p.z, mod);
            return p;
        }

        public Point3 ModPow2(Point3 modp) 
        {
            Point3 p = this;
            p.x = Mathf.ModPow2(p.x, modp.x);
            p.y = Mathf.ModPow2(p.y, modp.y);
            p.z = Mathf.ModPow2(p.z, modp.z);
            return p;
        }

        public Point3 ModBit2(int mod) 
        {
            Point3 p = this;
            p.x = Mathf.ModBit2(p.x, mod);
            p.y = Mathf.ModBit2(p.y, mod);
            p.z = Mathf.ModBit2(p.z, mod);
            return p;
        }

        public Point3 ModBit2(Point3 modp) 
        {
            Point3 p = this;
            p.x = Mathf.ModBit2(p.x, modp.x);
            p.y = Mathf.ModBit2(p.y, modp.y);
            p.z = Mathf.ModBit2(p.z, modp.z);
            return p;
        }

        public Point3 Rem(int rem) 
        {
            Point3 p = this;
            p.x = Mathf.RemInt(p.x, rem);
            p.y = Mathf.RemInt(p.y, rem);
            p.z = Mathf.RemInt(p.z, rem);
            return p;
        }

        public Point3 Rem(Point3 remp) 
        {
            Point3 p = this;
            p.x = Mathf.RemInt(p.x, remp.x);
            p.y = Mathf.RemInt(p.y, remp.y);
            p.z = Mathf.RemInt(p.z, remp.z);
            return p;
        }

        public void Set(real_t x, real_t y, real_t z)
        {
            this.x = Mathf.RoundToInt(x);
            this.y = Mathf.RoundToInt(y);
            this.z = Mathf.RoundToInt(z);
        }
        public void Set(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public void Set(Point3 p)
        {
            this.x = p.x;
            this.y = p.y;
            this.z = p.z;
        }
        public void Set(Vector3 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
            this.z = Mathf.RoundToInt(v.z);
        }

        public Vector3 ToVector() {
            return new Vector3 (this);
        }
        
        private static readonly Point3 zero    = new Point3 (0, 0, 0);
        private static readonly Point3 one     = new Point3 (1, 1, 1);
        private static readonly Point3 negOne  = new Point3 (-1, -1, -1);
    
        private static readonly Point3 up      = new Point3 (0, 1, 0);
        private static readonly Point3 down    = new Point3 (0, -1, 0);
        private static readonly Point3 right   = new Point3 (1, 0, 0);
        private static readonly Point3 left    = new Point3 (-1, 0, 0);
        private static readonly Point3 forward = new Point3 (0, 0, -1);
        private static readonly Point3 back    = new Point3 (0, 0, 1);

        public static Point3 Zero    { get { return zero;    } }
        public static Point3 One     { get { return one;     } }
        public static Point3 NegOne  { get { return negOne;  } }
        
        public static Point3 Up      { get { return up;      } }
        public static Point3 Down    { get { return down;    } }
        public static Point3 Right   { get { return right;   } }
        public static Point3 Left    { get { return left;    } }
        public static Point3 Forward { get { return forward; } }
        public static Point3 Back    { get { return back;    } }

        // Constructors
        public Point3(real_t x, real_t y, real_t z)
        {
            this.x = Mathf.RoundToInt(x);
            this.y = Mathf.RoundToInt(y);
            this.z = Mathf.RoundToInt(z);
        }
        public Point3(int x, int y, int z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public Point3(Point3 p)
        {
            this.x = p.x;
            this.y = p.y;
            this.z = p.z;
        }
        public Point3(Vector3 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
            this.z = Mathf.RoundToInt(v.z);
        }

        public static Point3 operator +(Point3 left, Point3 right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        public static Point3 operator -(Point3 left, Point3 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        public static Point3 operator -(Point3 point)
        {
            point.x = -point.x;
            point.y = -point.y;
            point.z = -point.z;
            return point;
        }

        public static Point3 operator *(Point3 point, int scale)
        {
            point.x *= scale;
            point.y *= scale;
            point.z *= scale;
            return point;
        }

        public static Point3 operator *(int scale, Point3 point)
        {
            point.x *= scale;
            point.y *= scale;
            point.z *= scale;
            return point;
        }

        public static Point3 operator *(Point3 left, Point3 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        public static Point3 operator /(Point3 point, int scale)
        {
            point.x /= scale;
            point.y /= scale;
            point.z /= scale;
            return point;
        }

        public static Point3 operator /(Point3 left, Point3 right)
        {
            left.x /= right.x;
            left.y /= right.y;
            left.z /= right.z;
            return left;
        }

        public static Point3 operator %(Point3 point, int mod)
        {
            return point.Mod(mod);
        }

        public static Point3 operator %(Point3 left, Point3 right)
        {
            return left.Mod(right);
        }

        public static Point3 operator &(Point3 point, int mod)
        {
            return point.ModBit2(mod);
        }

        public static Point3 operator &(Point3 point, Point3 mod)
        {
            return point.ModBit2(mod);
        }

        public static bool operator ==(Point3 left, Point3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Point3 left, Point3 right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Point3 left, Point3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z < right.z;
                else
                    return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >(Point3 left, Point3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z > right.z;
                else
                    return left.y > right.y;
            }

            return left.x > right.x;
        }

        public static bool operator <=(Point3 left, Point3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z <= right.z;
                else
                    return left.y < right.y;
            }

            return left.x < right.x;
        }

        public static bool operator >=(Point3 left, Point3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                    return left.z >= right.z;
                else
                    return left.y > right.y;
            }

            return left.x > right.x;
        }

        public override bool Equals(object obj)
        {
            if (obj is Point3)
            {
                return Equals((Point3)obj);
            }

            return false;
        }

        public bool Equals(Point3 other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode();
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



