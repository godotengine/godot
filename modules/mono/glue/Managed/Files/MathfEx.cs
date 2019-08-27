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

        public const real_t E = (real_t) 2.7182818284590452353602874714M; // 2.7182817f and 2.718281828459045
        public const real_t Sqrt2 = (real_t) 1.4142135623730950488016887242M; // 1.4142136f and 1.414213562373095

#if REAL_T_IS_DOUBLE
        public const real_t Epsilon = 1e-14; // Epsilon size should depend on the precision used.
#else
        public const real_t Epsilon = 1e-06f;
#endif

        public static int DecimalCount(real_t s)
        {
            return DecimalCount((decimal)s);
        }

        public static int DecimalCount(decimal s)
        {
            return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
        }

        public static int CeilToInt(real_t s)
        {
            return (int)Math.Ceiling(s);
        }

        public static int FloorToInt(real_t s)
        {
            return (int)Math.Floor(s);
        }

        public static int RoundToInt(real_t s)
        {
            return (int)Math.Round(s);
        }

        public static bool IsEqualApprox(real_t a, real_t b, real_t tolerance)
        {
            return Abs(a - b) < tolerance;
        }
    }
}