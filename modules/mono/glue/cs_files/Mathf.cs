using System;

#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    public static class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float. 
        public const real_t PI = (real_t) 3.1415926535897932384626433833M; // 3.1415927f and 3.14159265358979

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

        public static real_t Ceil(real_t s)
        {
            return (real_t)Math.Ceiling(s);
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

        public static real_t Fposmod(real_t x, real_t y)
        {
            if (x >= 0f)
            {
                return x % y;
            }
            else
            {
                return y - (-x % y);
            }
        }

        public static real_t Lerp(real_t from, real_t to, real_t weight)
        {
            return from + (to - from) * Clamp(weight, 0f, 1f);
        }

        public static real_t Log(real_t s)
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

        public static real_t Pow(real_t x, real_t y)
        {
            return (real_t)Math.Pow(x, y);
        }

        public static real_t Rad2Deg(real_t rad)
        {
            return rad * Rad2DegConst;
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

        public static int Wrap(int val, int min, int max)
        {
            int rng = max - min;
            return min + ((((val - min) % rng) + rng) % rng);
        }

        public static real_t Wrap(real_t val, real_t min, real_t max)
        {
            real_t rng = max - min;
            return min + (val - min) - (rng * Floor((val - min) / rng));
        }
    }
}
