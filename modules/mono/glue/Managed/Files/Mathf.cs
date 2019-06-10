using System;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    public static partial class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float.

        public const real_t Tau = (real_t) 6.2831853071795864769252867666M; // 6.2831855f and 6.28318530717959
        public const real_t Pi = (real_t) 3.1415926535897932384626433833M; // 3.1415927f and 3.14159265358979
        public const real_t Inf = real_t.PositiveInfinity;
        public const real_t NaN = real_t.NaN;

        private const real_t Deg2RadConst = (real_t) 0.0174532925199432957692369077M; // 0.0174532924f and 0.0174532925199433
        private const real_t Rad2DegConst = (real_t) 57.295779513082320876798154814M; // 57.29578f and 57.2957795130823

        public static real_t Abs(real_t s)
        {
            return Math.Abs(s);
        }

        public static int Abs(int s)
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

        public static real_t Atan2(real_t y, real_t x)
        {
            return (real_t)Math.Atan2(y, x);
        }

        public static Vector2 Cartesian2Polar(real_t x, real_t y)
        {
            return new Vector2(Sqrt(x * x + y * y), Atan2(y, x));
        }

        public static real_t Ceil(real_t s)
        {
            return (real_t)Math.Ceiling(s);
        }

        public static int Clamp(int value, int min, int max)
        {
            return value < min ? min : value > max ? max : value;
        }

        public static real_t Clamp(real_t value, real_t min, real_t max)
        {
            return value < min ? min : value > max ? max : value;
        }

        public static real_t Cos(real_t s)
        {
            return (real_t)Math.Cos(s);
        }

        public static real_t Cosh(real_t s)
        {
            return (real_t)Math.Cosh(s);
        }

        public static int StepDecimals(real_t step)
        {
            double[] sd = new double[] {
                0.9999,
                0.09999,
                0.009999,
                0.0009999,
                0.00009999,
                0.000009999,
                0.0000009999,
                0.00000009999,
                0.000000009999,
            };
            double abs = Mathf.Abs(step);
            double decs = abs - (int)abs; // Strip away integer part
            for (int i = 0; i < sd.Length; i++) {
                if (decs >= sd[i]) {
                    return i;
                }
            }
            return 0;
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

            if (curve < 0f)
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

        public static real_t InverseLerp(real_t from, real_t to, real_t weight)
        {
           return (weight - from) / (to - from);
        }

        public static bool IsEqualApprox(real_t a, real_t b)
        {
            real_t tolerance = Epsilon * Abs(a);
            if (tolerance < Epsilon) {
                tolerance = Epsilon;
            }
            return Abs(a - b) < tolerance;
        }

        public static bool IsInf(real_t s)
        {
           return real_t.IsInfinity(s);
        }

        public static bool IsNaN(real_t s)
        {
           return real_t.IsNaN(s);
        }

        public static bool IsZeroApprox(real_t s)
        {
            return Abs(s) < Epsilon;
        }

        public static real_t Lerp(real_t from, real_t to, real_t weight)
        {
            return from + (to - from) * weight;
        }

        public static real_t Log(real_t s)
        {
            return (real_t)Math.Log(s);
        }

        public static int Max(int a, int b)
        {
            return a > b ? a : b;
        }

        public static real_t Max(real_t a, real_t b)
        {
            return a > b ? a : b;
        }

        public static int Min(int a, int b)
        {
            return a < b ? a : b;
        }

        public static real_t Min(real_t a, real_t b)
        {
            return a < b ? a : b;
        }

        public static real_t MoveToward(real_t from, real_t to, real_t delta)
        {
            return Abs(to - from) <= delta ? to : from + Sign(to - from) * delta;
        }

        public static int NearestPo2(int value)
        {
            value--;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            value++;
            return value;
        }

        public static Vector2 Polar2Cartesian(real_t r, real_t th)
        {
            return new Vector2(r * Cos(th), r * Sin(th));
        }

        /// <summary>
        /// Performs a canonical Modulus operation, where the output is on the range [0, b).
        /// </summary>
        public static real_t PosMod(real_t a, real_t b)
        {
            real_t c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <summary>
        /// Performs a canonical Modulus operation, where the output is on the range [0, b).
        /// </summary>
        public static int PosMod(int a, int b)
        {
            int c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
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

        public static int Sign(int s)
        {
            return s < 0 ? -1 : 1;
        }

        public static real_t Sign(real_t s)
        {
            return s < 0f ? -1f : 1f;
        }

        public static real_t Sin(real_t s)
        {
            return (real_t)Math.Sin(s);
        }

        public static real_t Sinh(real_t s)
        {
            return (real_t)Math.Sinh(s);
        }

        public static real_t SmoothStep(real_t from, real_t to, real_t weight)
        {
            if (IsEqualApprox(from, to))
            {
                return from;
            }
            real_t x = Clamp((weight - from) / (to - from), (real_t)0.0, (real_t)1.0);
            return x * x * (3 - 2 * x);
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

        public static int Wrap(int value, int min, int max)
        {
            int rng = max - min;
            return rng != 0 ? min + ((value - min) % rng + rng) % rng : min;
        }

        public static real_t Wrap(real_t value, real_t min, real_t max)
        {
            real_t rng = max - min;
            return !IsEqualApprox(rng, default(real_t)) ? min + ((value - min) % rng + rng) % rng : min;
        }
    }
}
