#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;

namespace Godot
{
    /// <summary>
    /// 提供常用数学函数的常量和静态方法。
    /// </summary>
    public static partial class Mathf
    {
        // Define constants with Decimal precision and cast down to double or float.

        /// <summary>
        /// 圆常数，单位圆的周长，单位为弧度。
        /// </summary>
        // 6.2831855f and 6.28318530717959
        public const real_t Tau = (real_t)6.2831853071795864769252867666M;

        /// <summary>
        /// 代表圆直径多少倍的常数
        /// 适合它的周边。 这相当于 <c>Mathf.Tau / 2</c>。
        /// </summary>
        // 3.1415927f and 3.14159265358979
        public const real_t Pi = (real_t)3.1415926535897932384626433833M;

        /// <summary>
        /// 正无穷大。 对于负无穷大，使用 <c>-Mathf.Inf</c>。
        /// </summary>
        public const real_t Inf = real_t.PositiveInfinity;

        /// <summary>
        /// “不是数字”，无效值。 <c>NaN</c> 具有特殊属性，包括
        /// 它不等于自身。 它是由一些无效操作输出的，
        /// 比如将零除以零。
        /// </summary>
        public const real_t NaN = real_t.NaN;

        // 0.0174532924f and 0.0174532925199433
        private const real_t _deg2RadConst = (real_t)0.0174532925199432957692369077M;
        // 57.29578f and 57.2957795130823
        private const real_t _rad2DegConst = (real_t)57.295779513082320876798154814M;

        /// <summary>
        /// 返回 <paramref name="s"/> 的绝对值（即正值）。
        /// </summary>
        /// <param name="s">输入的数字。</param>
        /// <returns><paramref name="s"/>的绝对值。</returns>
        public static int Abs(int s)
        {
            return Math.Abs(s);
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 的绝对值（即正值）。
        /// </summary>
        /// <param name="s">输入的数字。</param>
        /// <returns><paramref name="s"/>的绝对值。</returns>
        public static real_t Abs(real_t s)
        {
            return Math.Abs(s);
        }

        /// <summary>
        /// 以弧度返回 <paramref name="s"/> 的反余弦值。
        /// 用于获取余弦角度 <paramref name="s"/>.
        /// </summary>
        /// <param name="s">输入余弦值。 必须在 -1.0 到 1.0 的范围内。</param>
        /// <returns>
        /// 将产生给定余弦值的角度。 在 <c>0</c> 到 <c>Tau/2</c> 的范围内。
        /// </returns>
        public static real_t Acos(real_t s)
        {
            return (real_t)Math.Acos(s);
        }

        /// <summary>
        /// 以弧度返回 <paramref name="s"/> 的反正弦值。
        /// 用于获取正弦角 <paramref name="s"/>.
        /// </summary>
        /// <param name="s">输入正弦值。 必须在 -1.0 到 1.0 的范围内。</param>
        /// <returns>
        /// 将产生给定正弦值的角度。 在 <c>-Tau/4</c> 到 <c>Tau/4</c> 范围内。
        /// </returns>
        public static real_t Asin(real_t s)
        {
            return (real_t)Math.Asin(s);
        }

        /// <summary>
        /// 以弧度返回 <paramref name="s"/> 的反正切。
        /// 用于获取切线的角度 <paramref name="s"/>。
        ///
        /// 该方法无法知道角度应该落在哪个象限。
        /// 如果您同时拥有 <c>y</c> 和 <c>x</c>，请参见 <see cref="Atan2(real_t, real_t)"/>。
        /// </summary>
        /// <param name="s">输入正切值。</param>
        /// <返回>
        /// 将导致给定正切值的角度。 在 <c>-Tau/4</c> 到 <c>Tau/4</c> 范围内。
        /// </returns>
        public static real_t Atan(real_t s)
        {
            return (real_t)Math.Atan(s);
        }

        /// <summary>
        /// 以弧度返回 <paramref name="y"/> 和 <paramref name="x"/> 的反正切。
        /// 用于获取<c>y/x</c>的切线角度。 为了计算该值，该方法采用
        /// 考虑两个参数的符号以确定象限。
        ///
        /// 重要提示：按照惯例，Y 坐标在前。
        /// </summary>
        /// <param name="y">要找到角度的点的Y坐标。</param>
        /// <param name="x">要找到角度的点的X坐标。</param>
        /// <returns>
        /// 将导致给定正切值的角度。 在 <c>-Tau/2</c> 到 <c>Tau/2</c> 范围内。
        /// </returns>
        public static real_t Atan2(real_t y, real_t x)
        {
            return (real_t)Math.Atan2(y, x);
        }

        /// <summary>
        /// 转换以笛卡尔坐标表示的二维点
        /// 系统（X和Y轴）到极坐标系
        ///（到原点的距离和角度）。
        /// </summary>
        /// <param name="x">输入的X坐标。</param>
        /// <param name="y">输入的Y坐标。</param>
        /// <returns>一个<see cref="Vector2"/>，其中X代表距离，Y代表角度。</returns>
        public static Vector2 Cartesian2Polar(real_t x, real_t y)
        {
            return new Vector2(Sqrt(x * x + y * y), Atan2(y, x));
        }

        /// <summary>
        /// 向上舍入 <paramref name="s"/>（朝向正无穷大）。
        /// </summary>
        /// <param name="s">上限的数字。</param>
        /// <returns>不小于<paramref name="s"/>的最小整数。</returns>
        public static real_t Ceil(real_t s)
        {
            return (real_t)Math.Ceiling(s);
        }

        /// <summary>
        /// 钳制一个 <paramref name="value"/> 使其不小于 <paramref name="min"/>
        /// 并且不超过 <paramref name="max"/>。
        /// </summary>
        /// <param name="value">要钳位的值。</param>
        /// <param name="min">允许的最小值。</param>
        /// <param name="max">最大允许值。</param>
        /// <returns>钳位值。</returns>
        public static int Clamp(int value, int min, int max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <summary>
        /// 钳制一个 <paramref name="value"/> 使其不小于 <paramref name="min"/>
        /// 并且不超过 <paramref name="max"/>。
        /// </summary>
        /// <param name="value">要钳位的值。</param>
        /// <param name="min">允许的最小值。</param>
        /// <param name="max">最大允许值。</param>
        /// <returns>钳位值。</returns>
        public static real_t Clamp(real_t value, real_t min, real_t max)
        {
            return value < min ? min : value > max ? max : value;
        }

        /// <summary>
        /// 以弧度返回角度 <paramref name="s"/> 的余弦值。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>那个角度的余弦值。</returns>
        public static real_t Cos(real_t s)
        {
            return (real_t)Math.Cos(s);
        }

        /// <summary>
        /// 以弧度返回角度 <paramref name="s"/> 的双曲余弦值。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>该角度的双曲余弦值。</returns>
        public static real_t Cosh(real_t s)
        {
            return (real_t)Math.Cosh(s);
        }

        /// <summary>
        /// 将以度数表示的角度转换为弧度。
        /// </summary>
        /// <param name="deg">以度数表示的角度。</param>
        /// <returns>以弧度表示的相同角度。</returns>
        public static real_t Deg2Rad(real_t deg)
        {
            return deg * _deg2RadConst;
        }

        /// <summary>
        /// 缓动函数，基于指数。 <paramref name="curve"/> 值是：
        /// <c>0</c>是常数，<c>1</c>是线性的，<c>0</c>到<c>1</c>是缓入的，<c> 1</c> 或更多是缓出。
        /// 负值是 in-out/out-in。
        /// </summary>
        /// <param name="s">缓动的值。</param>
        /// <param name="curve">
        /// <c>0</c>是常数，<c>1</c>是线性的，<c>0</c>到<c>1</c>是缓入的，<c> 1</c> 或更多是缓出。
        /// </参数>
        /// <returns>缓动值。</returns>
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

                return ((1.0f - Pow(1.0f - ((s - 0.5f) * 2.0f), -curve)) * 0.5f) + 0.5f;
            }

            return 0f;
        }

        /// <summary>
        /// 自然指数函数。 它提高了数学
        /// 常量 <c>e</c> 的 <paramref name="s"/> 次幂并返回它。
        /// </summary>
        /// <param name="s">将 <c>e</c> 提高到的指数。</param>
        /// <returns><c>e</c> 提升到 <paramref name="s"/> 的幂。</returns>
        public static real_t Exp(real_t s)
        {
            return (real_t)Math.Exp(s);
        }

        /// <summary>
        /// 向下舍入 <paramref name="s"/>（朝向负无穷大）。
        /// </summary>
        /// <param name="s">楼层数。</param>
        /// <returns>不大于<paramref name="s"/>的最大整数。</returns>
        public static real_t Floor(real_t s)
        {
            return (real_t)Math.Floor(s);
        }

        /// <summary>
        /// 考虑给定范围返回一个标准化值。
        /// 这与 <see cref="Lerp(real_t, real_t, real_t)"/> 相反。
        /// </summary>
        /// <param name="from">内插值。</param>
        /// <param name="to">插值的目标值。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>反插值的结果。</returns>
        public static real_t InverseLerp(real_t from, real_t to, real_t weight)
        {
            return (weight - from) / (to - from);
        }

        /// <summary>
        /// 如果 <paramref name="a"/> 和 <paramref name="b"/> 近似相等，则返回 <see langword="true"/>
        /// 对彼此。
        /// 比较是通过<see cref="Epsilon"/> 使用容差计算完成的。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <returns>A <see langword="bool"/> 两个值是否近似相等。</returns>
        public static bool IsEqualApprox(real_t a, real_t b)
        {
            // Check for exact equality first, required to handle "infinity" values.
            if (a == b)
            {
                return true;
            }
            // Then check for approximate equality.
            real_t tolerance = Epsilon * Abs(a);
            if (tolerance < Epsilon)
            {
                tolerance = Epsilon;
            }
            return Abs(a - b) < tolerance;
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 是否为无穷大值（正无穷大或负无穷大）。
        /// </summary>
        /// <param name="s">要检查的值。</param>
        /// <returns>一个 <see langword="bool"/> 该值是否为无穷大。</returns>
        public static bool IsInf(real_t s)
        {
            return real_t.IsInfinity(s);
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 是否为 <c>NaN</c>（“不是数字”或无效）值。
        /// </summary>
        /// <param name="s">要检查的值。</param>
        /// <returns>一个 <see langword="bool"/> 该值是否为 <c>NaN</c> 值。</returns>
        public static bool IsNaN(real_t s)
        {
            return real_t.IsNaN(s);
        }

        /// <summary>
        /// 如果 <paramref name="s"/> 近似为零，则返回 <see langword="true"/>。
        /// 比较是通过<see cref="Epsilon"/> 使用容差计算完成的。
        ///
        /// 此方法比使用一个值为零的 <see cref="IsEqualApprox(real_t, real_t)"/> 更快。
        /// </summary>
        /// <param name="s">要检查的值。</param>
        /// <returns>一个 <see langword="bool"/> 该值是否接近于零。</returns>
        public static bool IsZeroApprox(real_t s)
        {
            return Abs(s) < Epsilon;
        }

        /// <summary>
        /// 通过标准化值在两个值之间进行线性插值。
        /// 这是相反的<see cref="InverseLerp(real_t, real_t, real_t)"/>。
        /// </summary>
        /// <param name="from">插值的起始值。</param>
        /// <param name="to">插值的目标值。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果。</returns>
        public static real_t Lerp(real_t from, real_t to, real_t weight)
        {
            return from + ((to - from) * weight);
        }

        /// <summary>
        /// 通过归一化值在两个角度（以弧度）之间进行线性插值。
        ///
        /// 类似于 <see cref="Lerp(real_t, real_t, real_t)"/>,
        /// 但当角度环绕 <see cref="Tau"/> 时会正确插值。
        /// </summary>
        /// <param name="from">插值的起始角度。</param>
        /// <param name="to">插值的目标角度。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果角度。</returns>
        public static real_t LerpAngle(real_t from, real_t to, real_t weight)
        {
            real_t difference = (to - from) % Mathf.Tau;
            real_t distance = ((2 * difference) % Mathf.Tau) - difference;
            return from + (distance * weight);
        }

        /// <summary>
        /// 自然对数。 达到一定水平的持续增长所需的时间。
        ///
        /// 注意：这与大多数计算器上的“log”函数不同，后者使用以 10 为底的对数。
        /// </summary>
        /// <param name="s">输入值。</param>
        /// <returns><paramref name="s"/>的自然对数。</returns>
        public static real_t Log(real_t s)
        {
            return (real_t)Math.Log(s);
        }

        /// <summary>
        /// 返回两个值的最大值。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <returns>这两个值哪个更高。</returns>
        public static int Max(int a, int b)
        {
            return a > b ? a : b;
        }

        /// <summary>
        /// 返回两个值的最大值。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <returns>这两个值哪个更高。</returns>
        public static real_t Max(real_t a, real_t b)
        {
            return a > b ? a : b;
        }

        /// <summary>
        /// 返回两个值中的最小值。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <returns>两个值中的哪个值较小。</returns>
        public static int Min(int a, int b)
        {
            return a < b ? a : b;
        }

        /// <summary>
        /// 返回两个值中的最小值。
        /// </summary>
        /// <param name="a">其中一个值。</param>
        /// <param name="b">另一个值。</param>
        /// <returns>两个值中的哪个值较小。</returns>
        public static real_t Min(real_t a, real_t b)
        {
            return a < b ? a : b;
        }

        /// <summary>
        /// 将 <paramref name="from"/> 向 <paramref name="to"/> 移动 <paramref name="delta"/> 值。
        ///
        /// 使用负的 <paramref name="delta"/> 值移开。
        /// </summary>
        /// <param name="from">起始值。</param>
        /// <param name="to">要移动的值。</param>
        /// <param name="delta">移动量。</param>
        /// <returns>移动后的值</returns>
        public static real_t MoveToward(real_t from, real_t to, real_t delta)
        {
            if (Abs(to - from) <= delta)
                return to;

            return from + (Sign(to - from) * delta);
        }

        /// <summary>
        /// 返回整数 <paramref name="value"/> 最接近的 2 的较大幂。
        /// </summary>
        /// <param name="value">输入值。</param>
        /// <returns>最接近的 2 的较大幂。</returns>
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

        /// <summary>
        /// 转换以极坐标表示的二维点
        /// 系统（与原点的距离 <paramref name="r"/>
        /// 和一个角度 <paramref name="th"/>) 到笛卡尔
        /// 坐标系（X 和 Y 轴）。
        /// </summary>
        /// <param name="r">到原点的距离</param>
        /// <param name="th">点的角度。</param>
        /// <returns>代表笛卡尔坐标的<see cref="Vector2"/>。</returns>
        public static Vector2 Polar2Cartesian(real_t r, real_t th)
        {
            return new Vector2(r * Cos(th), r * Sin(th));
        }

        /// <summary>
        /// 执行规范模运算，其中输出在 [0, <paramref name="b"/>) 范围内。
        /// </summary>
        /// <param name="a">被除数，主要输入。</param>
        /// <param name="b">除数。 输出在 [0, <paramref name="b"/>) 范围内。</param>
        /// <returns>结果输出。</returns>
        public static int PosMod(int a, int b)
        {
            int c = a % b;
            if ((c < 0 && b > 0) || (c > 0 && b < 0))
            {
                c += b;
            }
            return c;
        }

        /// <summary>
        /// 执行规范模运算，其中输出在 [0, <paramref name="b"/>) 范围内。
        /// </summary>
        /// <param name="a">被除数，主要输入。</param>
        /// <param name="b">除数。 输出在 [0, <paramref name="b"/>) 范围内。</param>
        /// <returns>结果输出。</returns>
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
        /// 返回 <paramref name="x"/> 的 <paramref name="y"/> 次方的结果。
        /// </summary>
        /// <param name="x">基数。</param>
        /// <param name="y">指数。</param>
        /// <returns><paramref name="x"/> 提高到 <paramref name="y"/> 的幂。</returns>
        public static real_t Pow(real_t x, real_t y)
        {
            return (real_t)Math.Pow(x, y);
        }

        /// <summary>
        /// 将以弧度表示的角度转换为度数。
        /// </summary>
        /// <param name="rad">以弧度表示的角度。</param>
        /// <returns>以度数表示的相同角度。</returns>
        public static real_t Rad2Deg(real_t rad)
        {
            return rad * _rad2DegConst;
        }

        /// <summary>
        /// 将 <paramref name="s"/> 舍入到最接近的整数，
        /// 中间的情况向最接近的二的倍数舍入。
        /// </summary>
        /// <param name="s">要四舍五入的数字。</param>
        /// <returns>四舍五入的数字。</returns>
        public static real_t Round(real_t s)
        {
            return (real_t)Math.Round(s);
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 的符号：<c>-1</c> 或 <c>1</c>。
        /// 如果 <paramref name="s"/> 是 <c>0</c>，则返回 <c>0</c>。
        /// </summary>
        /// <param name="s">输入的数字。</param>
        /// <returns>三个可能值之一：<c>1</c>、<c>-1</c> 或 <c>0</c>。</returns>
        public static int Sign(int s)
        {
            if (s == 0)
                return 0;
            return s < 0 ? -1 : 1;
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 的符号：<c>-1</c> 或 <c>1</c>。
        /// 如果 <paramref name="s"/> 是 <c>0</c>，则返回 <c>0</c>。
        /// </summary>
        /// <param name="s">输入的数字。</param>
        /// <returns>三个可能值之一：<c>1</c>、<c>-1</c> 或 <c>0</c>。</returns>
        public static int Sign(real_t s)
        {
            if (s == 0)
                return 0;
            return s < 0 ? -1 : 1;
        }

        /// <summary>
        /// 以弧度返回角度 <paramref name="s"/> 的正弦值。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>那个角度的正弦值。</returns>
        public static real_t Sin(real_t s)
        {
            return (real_t)Math.Sin(s);
        }

        /// <summary>
        /// 以弧度返回角度 <paramref name="s"/> 的双曲正弦值。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>该角度的双曲正弦值。</returns>
        public static real_t Sinh(real_t s)
        {
            return (real_t)Math.Sinh(s);
        }

        /// <summary>
        /// 返回一个在 <paramref name="from"/> 和 <paramref name="to"/> 之间平滑插值的数字，
        /// 基于 <paramref name="weight"/>。 类似于 <see cref="Lerp(real_t, real_t, real_t)"/>，
        /// 但在开始时插值更快，最后插值更慢。
        /// </summary>
        /// <param name="from">插值的起始值。</param>
        /// <param name="to">插值的目标值。</param>
        /// <param name="weight">一个代表插值量的值。</param>
        /// <returns>插值的结果。</returns>
        public static real_t SmoothStep(real_t from, real_t to, real_t weight)
        {
            if (IsEqualApprox(from, to))
            {
                return from;
            }
            real_t x = Clamp((weight - from) / (to - from), (real_t)0.0, (real_t)1.0);
            return x * x * (3 - (2 * x));
        }

        /// <summary>
        /// 返回 <paramref name="s"/> 的平方根，其中 <paramref name="s"/> 是一个非负数。
        ///
        /// 如果您需要负输入，请使用 <see cref="System.Numerics.Complex"/>。
        /// </summary>
        /// <param name="s">输入数字。 不能为负数。</param>
        /// <returns><paramref name="s"/>的平方根。</returns>
        public static real_t Sqrt(real_t s)
        {
            return (real_t)Math.Sqrt(s);
        }

        /// <summary>
        /// 返回第一个非零数字的位置，在
        /// 小数点。 注意最大返回值为10，
        /// 这是实现中的设计决策。
        /// </summary>
        /// <param name="step">输入值。</param>
        /// <returns>第一个非零数字的位置。</returns>
        public static int StepDecimals(real_t step)
        {
            double[] sd = new double[]
            {
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
            double abs = Abs(step);
            double decs = abs - (int)abs; // Strip away integer part
            for (int i = 0; i < sd.Length; i++)
            {
                if (decs >= sd[i])
                {
                    return i;
                }
            }
            return 0;
        }

        /// <summary>
        /// 将浮点值 <paramref name="s"/> 捕捉到给定的 <paramref name="step"/>。
        /// 这也可用于将浮点数四舍五入为任意小数位数。
        /// </summary>
        /// <param name="s">要逐步化的值。</param>
        /// <param name="step">捕捉到的步长。</param>
        /// <returns></returns>
        public static real_t Stepify(real_t s, real_t step)
        {
            if (step != 0f)
            {
                return Floor((s / step) + 0.5f) * step;
            }

            return s;
        }

        /// <summary>
        /// 以弧度返回角度 <paramref name="s"/> 的正切。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>那个角度的正切。</returns>
        public static real_t Tan(real_t s)
        {
            return (real_t)Math.Tan(s);
        }

        /// <summary>
        /// 返回角度 <paramref name="s"/> 的双曲正切，单位为弧度。
        /// </summary>
        /// <param name="s">以弧度为单位的角度。</param>
        /// <returns>该角度的双曲正切。</returns>
        public static real_t Tanh(real_t s)
        {
            return (real_t)Math.Tanh(s);
        }

        /// <summary>
        /// 在 <paramref name="min"/> 和 <paramref name="max"/> 之间包裹 <paramref name="value"/>。
        /// 可用于创建类似循环的行为或无限曲面。
        /// 如果 <paramref name="min"/> 是 <c>0</c>，这是等价的
        /// 到 <see cref="PosMod(int, int)"/>，所以更喜欢使用它。
        /// </summary>
        /// <param name="value">要包装的值。</param>
        /// <param name="min">范围的最小值和下限。</param>
        /// <param name="max">范围的最大允许值和上限。</param>
        /// <returns>包装后的值。</returns>
        public static int Wrap(int value, int min, int max)
        {
            int range = max - min;
            if (range == 0)
                return min;

            return min + ((((value - min) % range) + range) % range);
        }

        /// <summary>
        /// 在 <paramref name="min"/> 和 <paramref name="max"/> 之间包裹 <paramref name="value"/>。
        /// 可用于创建类似循环的行为或无限曲面。
        /// 如果 <paramref name="min"/> 是 <c>0</c>，这是等价的
        /// 到 <see cref="PosMod(real_t, real_t)"/>，所以更喜欢使用它。
        /// </summary>
        /// <param name="value">要包装的值。</param>
        /// <param name="min">范围的最小值和下限。</param>
        /// <param name="max">范围的最大允许值和上限。</param>
        /// <returns>包装后的值。</returns>
        public static real_t Wrap(real_t value, real_t min, real_t max)
        {
            real_t range = max - min;
            if (IsZeroApprox(range))
                return min;

            return min + ((((value - min) % range) + range) % range);
        }
    }
}
