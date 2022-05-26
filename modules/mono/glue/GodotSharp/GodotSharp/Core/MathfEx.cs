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
        /// 自然数<c>e</c>。
        /// </summary>
        public const real_t E = (real_t)2.7182818284590452353602874714M; // 2.7182817f and 2.718281828459045

        /// <summary>
        /// 2 的平方根。
        /// </summary>
        public const real_t Sqrt2 = (real_t)1.4142135623730950488016887242M; // 1.4142136f and 1.414213562373095

        /// <summary>
        /// 一个非常小的数字，用于具有容错性的浮点比较。
        /// 1e-06 具有单精度浮点数，但 1e-14 如果 <c>REAL_T_IS_DOUBLE</c>。
        /// </summary>
#if REAL_T_IS_DOUBLE
        public const real_t Epsilon = 1e-14; // Epsilon size should depend on the precision used.
#else
        public const real_t Epsilon = 1e-06f;
#endif

        /// <summary>
        /// 返回小数点后的位数。
        /// </summary>
        /// <param name="s">输入值。</param>
        /// <returns>位数。</returns>
        public static int DecimalCount(real_t s)
        {
            return DecimalCount((decimal)s);
        }

        /// <summary>
        /// 返回小数点后的位数。
        /// </summary>
        /// <param name="s">输入的<see cref="decimal"/>值。</param>
        /// <returns>位数。</returns>
        public static int DecimalCount(decimal s)
        {
            return BitConverter.GetBytes(decimal.GetBits(s)[3])[2];
        }

        /// <summary>
        /// 向上舍入 <paramref name="s"/>（朝向正无穷大）。
        ///
        /// 这与 <see cref="Ceil(real_t)"/> 相同，但返回一个 <c>int</c>。
        /// </summary>
        /// <param name="s">上限的数字。</param>
        /// <returns>不小于<paramref name="s"/>的最小整数。</returns>
        public static int CeilToInt(real_t s)
        {
            return (int)Math.Ceiling(s);
        }

        /// <summary>
        /// 向下舍入 <paramref name="s"/>（朝向负无穷大）。
        ///
        /// 这与 <see cref="Floor(real_t)"/> 相同，但返回一个 <c>int</c>。
        /// </summary>
        /// <param name="s">楼层数。</param>
        /// <returns>不大于<paramref name="s"/>的最大整数。</returns>
        public static int FloorToInt(real_t s)
        {
            return (int)Math.Floor(s);
        }

        /// <summary>
        /// 将 <paramref name="s"/> 舍入到最接近的整数。
        ///
        /// 这与 <see cref="Round(real_t)"/> 相同，但返回一个 <c>int</c>。
        /// </summary>
        /// <param name="s">要四舍五入的数字。</param>
        /// <returns>四舍五入的数字。</returns>
        public static int RoundToInt(real_t s)
        {
            return (int)Math.Round(s);
        }

        /// <summary>
        /// 如果 <paramref name="a"/> 和 <paramref name="b"/> 近似，则返回 <see langword="true"/>
        ///彼此相等。
        /// 使用提供的容差值进行比较。
        /// 如果您想为您计算公差，请使用 <see cref="IsEqualApprox(real_t, real_t)"/>。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <param name="tolerance">预先计算的公差值。</param>
        /// <returns>A <see langword="bool"/> 两个值是否相等。</returns>
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
