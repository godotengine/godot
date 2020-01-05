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
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b) {
                return true;
            }
            // Then check for approximate equality.
            return Abs(a - b) < tolerance;
        }
    }
}
