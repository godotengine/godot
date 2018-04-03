using System;

#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    /// <summary>
    /// Mathf is Godot's wrapper class for mathematical functions.
    /// For various reasons, we recommend using Mathf instead of Math for your game.
    /// </summary>
    public static class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float. 
        public const real_t TAU = (real_t) 6.2831853071795864769252867666M; // 6.2831855f and 6.28318530717959
        public const real_t  PI = (real_t) 3.1415926535897932384626433833M; // 3.1415927f and 3.14159265358979
        public const real_t   E = (real_t) 2.7182818284590452353602874714M; // 2.7182817f and 2.718281828459045
        public const real_t RT2 = (real_t) 1.4142135623730950488016887242M; // 1.4142136f and 1.414213562373095

        #if REAL_T_IS_DOUBLE
        public const real_t Epsilon = 1e-14; // Epsilon size should depend on the precision used.
        #else
        public const real_t Epsilon = 1e-06f;
        #endif

        private const real_t Deg2RadConst = (real_t) 0.0174532925199432957692369077M; // 0.0174532924f and 0.0174532925199433
        private const real_t Rad2DegConst = (real_t) 57.295779513082320876798154814M; // 57.29578f and 57.2957795130823

        public static real_t Abs(real_t s)
        {
            return Math.Abs(s);
        }

        public static real_t Acos(real_t s)
        {
            return (real_t)Math.Acos(s);
        }

        public static real_t Asin(real_t s)
        {
            return (real_t)Math.Asin(s);
        }

        public static real_t Atan(real_t s)
        {
            return (real_t)Math.Atan(s);
        }

        public static real_t Atan2(real_t x, real_t y)
        {
            return (real_t)Math.Atan2(x, y);
        }

        public static Vector2 Cartesian2Polar(real_t x, real_t y)
        {
            return new Vector2(Sqrt(x * x + y * y), Atan2(y, x));
        }

        public static Vector2 Cartesian2Polar(Vector2 v)
        {
            return new Vector2(Sqrt(v.x * v.x + v.y * v.y), Atan2(v.y, v.x));
        }

        public static real_t Ceil(real_t s)
        {
            return (real_t)Math.Ceiling(s);
        }

        public static int CeilToInt(real_t s)
        {
            return (int)Math.Ceiling(s);
        }

        public static real_t Clamp(real_t val, real_t min, real_t max)
        {
            if (val < min)
            {
                return min;
            }
            else if (val > max)
            {
                return max;
            }

            return val;
        }

        public static real_t Cos(real_t s)
        {
            return (real_t)Math.Cos(s);
        }

        public static real_t Cosh(real_t s)
        {
            return (real_t)Math.Cosh(s);
        }

        public static int Decimals(real_t step)
        {
            return Decimals((decimal)step);
        }

        public static int Decimals(decimal step)
        {
            return BitConverter.GetBytes(decimal.GetBits(step)[3])[2];
        }

        public static real_t Deg2Rad(real_t deg)
        {
            return deg * Deg2RadConst;
        }

        /// <summary>
        /// This method performs floored division, equivalent to // in Python. 
        /// Ex: DivFloor(10, 7) is 1, while DivFloor(-10, 7) is -2.
        /// This is useful in conjunction with Modulus.
        /// </summary>
        public static int DivFloor(real_t a, real_t b) 
        {
            real_t c = a / b;
            return (int)Math.Floor(c);
        }

        public static real_t Ease(real_t s, real_t curve)
        {
            if (s < 0f)
            {
                s = 0f;
            }
            else if (s > 1.0f)
            {
                s = 1.0f;
            }

            if (curve > 0f)
            {
                if (curve < 1.0f)
                {
                    return 1.0f - Pow(1.0f - s, 1.0f / curve);
                }

                return Pow(s, curve);
            }
            else if (curve < 0f)
            {
                if (s < 0.5f)
                {
                    return Pow(s * 2.0f, -curve) * 0.5f;
                }

                return (1.0f - Pow(1.0f - (s - 0.5f) * 2.0f, -curve)) * 0.5f + 0.5f;
            }

            return 0f;
        }

        public static real_t Exp(real_t s)
        {
            return (real_t)Math.Exp(s);
        }

        public static real_t Floor(real_t s)
        {
            return (real_t)Math.Floor(s);
        }

        public static int FloorToInt(real_t s)
        {
            return (int)Math.Floor(s);
        }

        public static real_t Fposmod(real_t a, real_t b)
        {
            real_t c = a % b;
            if (c < 0) 
            {
                c += b;
            }
            return c;
        }

        public static real_t Lerp(real_t from, real_t to, real_t weight)
        {
            return from + (to - from) * Clamp(weight, 0f, 1f);
        }

        /// <summary>
        /// This returns the Logarithm with base E (Natural Log)
        /// </summary>
        public static real_t Log(real_t s)
        {
            return (real_t)Math.Log(s);
        }

        /// <summary>
        /// This returns the Logarithm with base `b` 
        /// </summary>
        public static real_t LogBase(real_t a, real_t b)
        {
            return (real_t)Math.Log(a, b);
        }

        public static real_t Log10(real_t s)
        {
            return (real_t)Math.Log10(s);
        }

        public static real_t Log2(real_t s)
        {
            return (real_t)Math.Log(s, 2);
        }

        public static real_t LogE(real_t s)
        {
            return (real_t)Math.Log(s);
        }

        public static int Max(int a, int b)
        {
            return (a > b) ? a : b;
        }

        public static real_t Max(real_t a, real_t b)
        {
            return (a > b) ? a : b;
        }

        public static int Min(int a, int b)
        {
            return (a < b) ? a : b;
        }

        public static real_t Min(real_t a, real_t b)
        {
            return (a < b) ? a : b;
        }

        public static real_t Mod(real_t a, real_t b)
        {
            real_t c = a % b;
            if (c < 0) 
            {
                c += b;
            }
            return c;
        }

        public static int ModInt(real_t a, real_t b)
        {
            real_t c = a % b;
            if (c < 0) 
            {
                c += b;
            }
            return RoundToInt(c);
        }

        public static int ModInt(int a, int b)
        {
            int c = a % b;
            if (c < 0) 
            {
                c += b;
            }
            return c;
        }

        /// <summary>
        /// Very fast Modulus function using a bitwise operation. 
        /// B is only valid for powers of two, such as 2, 4, 8, 16, etc.
        /// If you are looking for pure speed, see if ModBit2 works for you.
        /// </summary>
        public static int ModPow2(int a, int b)
        {
            return (a & (b-1));
        }

        /// <summary>
        /// Extremely fast Modulus function using a bitwise operation. 
        /// B is only valid for powers of two minus one, such as 1, 3, 7, 15, etc.
        /// Output is the same as ModPow2 with the corresponding power of two.
        /// For example, `ModBit2(a, 15)` is the same as `ModPow2(a, 16)`.
        /// </summary>
        public static int ModBit2(int a, int b)
        {
            return (a & b);
        }

        public static int NearestPo2(int val)
        {
            val--;
            val |= val >> 1;
            val |= val >> 2;
            val |= val >> 4;
            val |= val >> 8;
            val |= val >> 16;
            val++;
            return val;
        }

        public static Vector2 Polar2Cartesian(real_t r, real_t th)
        {
            return new Vector2(r * Cos(th), r * Sin(th));
        }

        public static Vector2 Polar2Cartesian(Vector2 v)
        {
            return new Vector2(v.x * Cos(v.y), v.x * Sin(v.y));
        }

        public static real_t Pow(real_t x, real_t y)
        {
            return (real_t)Math.Pow(x, y);
        }

        public static real_t Rad2Deg(real_t rad)
        {
            return rad * Rad2DegConst;
        }

        public static real_t Rem(real_t a, real_t b) 
        {
            return (a % b);
        }

        public static int RemInt(real_t a, real_t b) 
        {
            return RoundToInt(a % b);
        }

        public static int RemInt(int a, int b) 
        {
            return (a % b);
        }

        public static real_t Round(real_t s)
        {
            return (real_t)Math.Round(s);
        }

        public static int RoundToInt(real_t s)
        {
            return (int)Math.Round(s);
        }

        public static real_t Sign(real_t s)
        {
            return (s < 0f) ? -1f : 1f;
        }

        public static real_t Sin(real_t s)
        {
            return (real_t)Math.Sin(s);
        }

        public static real_t Sinh(real_t s)
        {
            return (real_t)Math.Sinh(s);
        }

        public static real_t Sqrt(real_t s)
        {
            return (real_t)Math.Sqrt(s);
        }

        public static real_t Stepify(real_t s, real_t step)
        {
            if (step != 0f)
            {
                s = Floor(s / step + 0.5f) * step;
            }

            return s;
        }

        public static real_t Tan(real_t s)
        {
            return (real_t)Math.Tan(s);
        }

        public static real_t Tanh(real_t s)
        {
            return (real_t)Math.Tanh(s);
        }
    }
}



