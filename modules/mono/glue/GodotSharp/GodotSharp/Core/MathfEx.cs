#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;

namespace Godot
{
    public static partial class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float.

        /// <summary>
        /// The natural number <c>e</c>.
        /// </summary>
        public const real_t E = (real_t)2.7182818284590452353602874714M; // 2.7182817f and 2.718281828459045

        /// <summary>
        /// The square root of 2.
        /// </summary>
        public const real_t Sqrt2 = (real_t)1.4142135623730950488016887242M; // 1.4142136f and 1.414213562373095

        /// <summary>
        /// A very small number used for float comparison with error tolerance.
        /// 1e-06 with single-precision floats, but 1e-14 if <c>REAL_T_IS_DOUBLE</c>.
        /// </summary>
#if REAL_T_IS_DOUBLE
        public const real_t Epsilon = 1e-14; // Epsilon size should depend on the precision used.
#else
        public const real_t Epsilon = 1e-06f;
#endif

        /// <summary>
        /// Returns the amount of digits after the decimal place.
        /// </summary>
        /// <param name="s">The input value.</param>
        /// <returns>The amount of digits.</returns>
        public static int DecimalCount(real_t s)
        {
            return DecimalCount((decimal)s);
        }

        /// <summary>
        /// Returns the amount of digits after the decimal place.
        /// </summary>
        /// <param name="s">The input <see cref="decimal"/> value.</param>
        /// <returns>The amount of digits.</returns>
        public static int DecimalCount(decimal s)
        {
            return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
        }

        /// <summary>
        /// Rounds <paramref name="s"/> upward (towards positive infinity).
        ///
        /// This is the same as <see cref="Ceil(real_t)"/>, but returns an <c>int</c>.
        /// </summary>
        /// <param name="s">The number to ceil.</param>
        /// <returns>The smallest whole number that is not less than <paramref name="s"/>.</returns>
        public static int CeilToInt(real_t s)
        {
            return (int)Math.Ceiling(s);
        }

        /// <summary>
        /// Rounds <paramref name="s"/> downward (towards negative infinity).
        ///
        /// This is the same as <see cref="Floor(real_t)"/>, but returns an <c>int</c>.
        /// </summary>
        /// <param name="s">The number to floor.</param>
        /// <returns>The largest whole number that is not more than <paramref name="s"/>.</returns>
        public static int FloorToInt(real_t s)
        {
            return (int)Math.Floor(s);
        }

        /// <summary>
        /// Rounds <paramref name="s"/> to the nearest whole number.
        ///
        /// This is the same as <see cref="Round(real_t)"/>, but returns an <c>int</c>.
        /// </summary>
        /// <param name="s">The number to round.</param>
        /// <returns>The rounded number.</returns>
        public static int RoundToInt(real_t s)
        {
            return (int)Math.Round(s);
        }

        /// <summary>
        /// Returns <see langword="true"/> if <paramref name="a"/> and <paramref name="b"/> are approximately
        /// equal to each other.
        /// The comparison is done using the provided tolerance value.
        /// If you want the tolerance to be calculated for you, use <see cref="IsEqualApprox(real_t, real_t)"/>.
        /// </summary>
        /// <param name="a">One of the values.</param>
        /// <param name="b">The other value.</param>
        /// <param name="tolerance">The pre-calculated tolerance value.</param>
        /// <returns>A <see langword="bool"/> for whether or not the two values are equal.</returns>
        public static bool IsEqualApprox(real_t a, real_t b, real_t tolerance)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            return Abs(a - b) < tolerance;
        }
    }
}
