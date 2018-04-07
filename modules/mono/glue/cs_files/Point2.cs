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
    /// Point2 is a class similar to Vector2, except with integers.
    /// It is useful, for example, when working with points on a 2D grid.
    /// Not all concepts of Vector2 apply, such as normalization and any functions that depend on normalization. 
    /// However, new concepts can exist with Point2 such as bitwise operations.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Point2 : IEquatable<Point2>
    {
        public enum Axis
        {
            X = 0,
            Y
        }

        public int x;
        public int y;

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

        public Point2 Abs()
        {
            return new Point2( Mathf.RoundToInt(Mathf.Abs(x)), Mathf.RoundToInt(Mathf.Abs(y)) );
        }

        public int DistanceSquaredTo(Point2 b)
        {
            return (b - this).LengthSquared();
        }

        public real_t DistanceTo(Point2 b)
        {
            return (b - this).Length();
        }

        public Point2 DivFloor(int div) 
        {
            Point2 p = this;
            p.x = Mathf.DivFloor(p.x, div);
            p.y = Mathf.DivFloor(p.y, div);
            return p;
        }

        public Point2 DivFloor(Point2 divp) 
        {
            Point2 p = this;
            p.x = Mathf.DivFloor(p.x, divp.x);
            p.y = Mathf.DivFloor(p.y, divp.y);
            return p;
        }

        public int Dot(Point2 b)
        {
            return x * b.x + y * b.y;
        }

        public real_t Length()
        {
            int x2 = x * x;
            int y2 = y * y;

            return Mathf.Sqrt(x2 + y2);
        }

        public int LengthSquared()
        {
            int x2 = x * x;
            int y2 = y * y;

            return x2 + y2;
        }

        public Axis MaxAxis()
        {
            return x < y ? Axis.Y : Axis.X;
        }

        public Axis MinAxis()
        {
            return x > y ? Axis.Y : Axis.X;
        }

        public Point2 Mod(int mod) 
        {
            Point2 p = this;
            p.x = Mathf.ModInt(p.x, mod);
            p.y = Mathf.ModInt(p.y, mod);
            return p;
        }

        public Point2 Mod(Point2 modp) 
        {
            Point2 p = this;
            p.x = Mathf.ModInt(p.x, modp.x);
            p.y = Mathf.ModInt(p.y, modp.y);
            return p;
        }

        public Point2 ModPow2(int mod) 
        {
            Point2 p = this;
            p.x = Mathf.ModPow2(p.x, mod);
            p.y = Mathf.ModPow2(p.y, mod);
            return p;
        }

        public Point2 ModPow2(Point2 modp) 
        {
            Point2 p = this;
            p.x = Mathf.ModPow2(p.x, modp.x);
            p.y = Mathf.ModPow2(p.y, modp.y);
            return p;
        }

        public Point2 ModBit2(int mod) 
        {
            Point2 p = this;
            p.x = Mathf.ModBit2(p.x, mod);
            p.y = Mathf.ModBit2(p.y, mod);
            return p;
        }

        public Point2 ModBit2(Point2 modp) 
        {
            Point2 p = this;
            p.x = Mathf.ModBit2(p.x, modp.x);
            p.y = Mathf.ModBit2(p.y, modp.y);
            return p;
        }

        public Point2 Rem(int rem) 
        {
            Point2 p = this;
            p.x = Mathf.RemInt(p.x, rem);
            p.y = Mathf.RemInt(p.y, rem);
            return p;
        }

        public Point2 Rem(Point2 remp) 
        {
            Point2 p = this;
            p.x = Mathf.RemInt(p.x, remp.x);
            p.y = Mathf.RemInt(p.y, remp.y);
            return p;
        }

        public void Set(real_t x, real_t y)
        {
            this.x = Mathf.RoundToInt(x);
            this.y = Mathf.RoundToInt(y);
        }
        public void Set(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
        public void Set(Point2 p)
        {
            this.x = p.x;
            this.y = p.y;
        }
        public void Set(Vector2 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
        }

        public Point2 Tangent()
        {
            return new Point2(y, -x);
        }

        public Vector2 ToVector() {
            return new Vector2 (this);
        }
        
        private static readonly Point2 zero    = new Point2 (0, 0);
        private static readonly Point2 one     = new Point2 (1, 1);
        private static readonly Point2 negOne  = new Point2 (-1, -1);
    
        private static readonly Point2 up      = new Point2 (0, 1);
        private static readonly Point2 down    = new Point2 (0, -1);
        private static readonly Point2 right   = new Point2 (1, 0);
        private static readonly Point2 left    = new Point2 (-1, 0);

        public static Point2 Zero    { get { return zero;    } }
        public static Point2 One     { get { return one;     } }
        public static Point2 NegOne  { get { return negOne;  } }
        
        public static Point2 Up      { get { return up;      } }
        public static Point2 Down    { get { return down;    } }
        public static Point2 Right   { get { return right;   } }
        public static Point2 Left    { get { return left;    } }

        // Constructors
        public Point2(real_t x, real_t y)
        {
            this.x = Mathf.RoundToInt(x);
            this.y = Mathf.RoundToInt(y);
        }
        public Point2(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
        public Point2(Point2 p)
        {
            this.x = p.x;
            this.y = p.y;
        }
        public Point2(Vector2 v)
        {
            this.x = Mathf.RoundToInt(v.x);
            this.y = Mathf.RoundToInt(v.y);
        }

        public static Point2 operator +(Point2 left, Point2 right)
        {
            left.x += right.x;
            left.y += right.y;
            return left;
        }

        public static Point2 operator -(Point2 left, Point2 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            return left;
        }

        public static Point2 operator -(Point2 point)
        {
            point.x = -point.x;
            point.y = -point.y;
            return point;
        }

        public static Point2 operator *(Point2 point, int scale)
        {
            point.x *= scale;
            point.y *= scale;
            return point;
        }

        public static Point2 operator *(int scale, Point2 point)
        {
            point.x *= scale;
            point.y *= scale;
            return point;
        }

        public static Point2 operator *(Point2 left, Point2 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            return left;
        }

        public static Point2 operator /(Point2 point, int scale)
        {
            point.x /= scale;
            point.y /= scale;
            return point;
        }

        public static Point2 operator /(Point2 left, Point2 right)
        {
            left.x /= right.x;
            left.y /= right.y;
            return left;
        }

        public static Point2 operator %(Point2 point, int mod)
        {
            return point.Mod(mod);
        }

        public static Point2 operator %(Point2 left, Point2 right)
        {
            return left.Mod(right);
        }

        public static Point2 operator &(Point2 point, int mod)
        {
            return point.ModBit2(mod);
        }

        public static Point2 operator &(Point2 point, Point2 mod)
        {
            return point.ModBit2(mod);
        }

        public static bool operator ==(Point2 left, Point2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Point2 left, Point2 right)
        {
            return !left.Equals(right);
        }

        public static bool operator <(Point2 left, Point2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        public static bool operator >(Point2 left, Point2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        public static bool operator <=(Point2 left, Point2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y <= right.y;
            }
            return left.x <= right.x;
        }

        public static bool operator >=(Point2 left, Point2 right)
        {
            if (left.x.Equals(right.x))
            {
                return left.y >= right.y;
            }
            return left.x >= right.x;
        }

        public override bool Equals(object obj)
        {
            if (obj is Point2)
            {
                return Equals((Point2)obj);
            }

            return false;
        }

        public bool Equals(Point2 other)
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



