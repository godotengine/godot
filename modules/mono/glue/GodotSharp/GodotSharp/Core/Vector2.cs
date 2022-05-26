#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Runtime.InteropServices;

namespace Godot
{
    /// <summary>
    /// 2 元素结构，可用于表示 2D 空间中的位置或任何其他数值对。
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector2 : IEquatable<Vector2>
    {
        /// <summary>
        /// 轴的枚举索引值。
        /// 由 <see cref="MaxAxis"/> 和 <see cref="MinAxis"/> 返回。
        /// </summary>
        public enum Axis
        {
            /// <summary>
            /// 向量的 X 轴。
            /// </summary>
            X = 0,
            /// <summary>
            /// 向量的 Y 轴。
            /// </summary>
            Y
        }

        /// <summary>
        /// 向量的 X 分量。 也可以通过使用索引位置 <c>[0]</c> 访问。
        /// </summary>
        public real_t x;

        /// <summary>
        /// 向量的 Y 分量。 也可以通过使用索引位置 <c>[1]</c> 访问。
        /// </summary>
        public real_t y;

        /// <summary>
        /// 使用它们的索引访问向量分量。
        /// </summary>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the given the <paramref name="index"/> is not 0 or 1.
        /// </exception>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="x"/>,
        /// <c>[1]</c> is equivalent to <see cref="y"/>.
        /// </value>
        public real_t this[int index]
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

        internal void Normalize()
        {
            real_t lengthsq = LengthSquared();

            if (lengthsq == 0)
            {
                x = y = 0f;
            }
            else
            {
                real_t length = Mathf.Sqrt(lengthsq);
                x /= length;
                y /= length;
            }
        }

        /// <summary>
        /// 返回一个新向量，其中所有分量均为绝对值（即正数）。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Abs(real_t)"/> 的向量。</returns>
        public Vector2 Abs()
        {
            return new Vector2(Mathf.Abs(x), Mathf.Abs(y));
        }

        /// <summary>
        /// 返回此向量相对于 X 轴的角度，或 (1, 0) 向量，单位为弧度。
        ///
        /// 等价于 <see cref="Mathf.Atan2(real_t, real_t)"/> 时的结果
        /// 使用向量的 <see cref="y"/> 和 <see cref="x"/> 作为参数调用：<c>Mathf.Atan2(v.y, v.x)</c>。
        /// </summary>
        /// <returns>这个向量的角度，以弧度为单位。</returns>
        public real_t Angle()
        {
            return Mathf.Atan2(y, x);
        }

        /// <summary>
        /// 返回给定向量的角度，以弧度为单位。
        /// </summary>
        /// <param name="to">与此向量进行比较的另一个向量。</param>
        /// <returns>两个向量之间的角度，以弧度为单位。</returns>
        public real_t AngleTo(Vector2 to)
        {
            return Mathf.Atan2(Cross(to), Dot(to));
        }

        /// <summary>
        /// 返回连接两点的直线与 X 轴之间的角度，以弧度为单位。
        /// </summary>
        /// <param name="to">与此向量进行比较的另一个向量。</param>
        /// <returns>两个向量之间的角度，以弧度为单位。</returns>
        public real_t AngleToPoint(Vector2 to)
        {
            return Mathf.Atan2(y - to.y, x - to.x);
        }

        /// <summary>
        /// 返回此向量的纵横比，<see cref="x"/> 与 <see cref="y"/> 的比率。
        /// </summary>
        /// <returns><see cref="x"/> 组件除以 <see cref="y"/> 组件。</returns>
        public real_t Aspect()
        {
            return x / y;
        }

        /// <summary>
        /// 返回从给定法线定义的平面“反弹”的向量。
        /// </summary>
        /// <param name="normal">定义要反弹的平面的法线向量。 必须标准化。</param>
        /// <returns>反弹的向量。</returns>
        public Vector2 Bounce(Vector2 normal)
        {
            return -Reflect(normal);
        }

        /// <summary>
        /// 返回一个所有分量向上舍入的新向量（朝向正无穷大）。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Ceil"/> 的向量。</returns>
        public Vector2 Ceil()
        {
            return new Vector2(Mathf.Ceil(x), Mathf.Ceil(y));
        }

        /// <summary>
        /// 通过将其长度限制为 <paramref name="length"/> 来返回具有最大长度的向量。
        /// </summary>
        /// <param name="length">要限制的长度。</param>
        /// <returns>长度有限的向量。</returns>
        [Obsolete("Clamped is deprecated because it has been renamed to LimitLength.")]
        public Vector2 Clamped(real_t length)
        {
            var v = this;
            real_t l = Length();

            if (l > 0 && length < l)
            {
                v /= l;
                v *= length;
            }

            return v;
        }

        /// <summary>
        /// 返回此向量与 <paramref name="b"/> 的叉积。
        /// </summary>
        /// <param name="b">另一个向量。</param>
        /// <returns>叉积值。</returns>
        public real_t Cross(Vector2 b)
        {
            return (x * b.y) - (y * b.x);
        }

        /// <summary>
        /// 在向量之间执行三次插值 <paramref name="preA"/>, 这个向量,
        /// <paramref name="b"/> 和 <paramref name="postB"/>，按给定的数量 <paramref name="weight"/>。
        /// </summary>
        /// <param name="b">目标向量。</param>
        /// <param name="preA">这个向量之前的一个向量。</param>
        /// <param name="postB"><paramref name="b"/>之后的一个向量。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值向量。</returns>
        public Vector2 CubicInterpolate(Vector2 b, Vector2 preA, Vector2 postB, real_t weight)
        {
            Vector2 p0 = preA;
            Vector2 p1 = this;
            Vector2 p2 = b;
            Vector2 p3 = postB;

            real_t t = weight;
            real_t t2 = t * t;
            real_t t3 = t2 * t;

            return 0.5f * (
                (p1 * 2.0f) +
                ((-p0 + p2) * t) +
                (((2.0f * p0) - (5.0f * p1) + (4 * p2) - p3) * t2) +
                ((-p0 + (3.0f * p1) - (3.0f * p2) + p3) * t3)
            );
        }

        /// <summary>
        /// 返回从这个向量指向 <paramref name="b"/> 的归一化向量。
        /// </summary>
        /// <param name="b">另一个指向的向量。</param>
        /// <returns>从这个向量到<paramref name="b"/>的方向。</returns>
        public Vector2 DirectionTo(Vector2 b)
        {
            return new Vector2(b.x - x, b.y - y).Normalized();
        }

        /// <summary>
        /// 返回此向量与 <paramref name="to"/> 之间的平方距离。
        /// 这个方法比 <see cref="DistanceTo"/> 运行得快，所以如果
        /// 你需要比较向量或者需要一些公式的平方距离。
        /// </summary>
        /// <param name="to">要使用的另一个向量。</param>
        /// <returns>两个向量之间的平方距离。</returns>
        public real_t DistanceSquaredTo(Vector2 to)
        {
            return (x - to.x) * (x - to.x) + (y - to.y) * (y - to.y);
        }

        /// <summary>
        /// 返回此向量与 <paramref name="to"/> 之间的距离。
        /// </summary>
        /// <param name="to">要使用的另一个向量。</param>
        /// <returns>两个向量之间的距离。</returns>
        public real_t DistanceTo(Vector2 to)
        {
            return Mathf.Sqrt((x - to.x) * (x - to.x) + (y - to.y) * (y - to.y));
        }

        /// <summary>
        /// 返回此向量与 <paramref name="with"/> 的点积。
        /// </summary>
        /// <param name="with">另一个要使用的向量。</param>
        /// <returns>两个向量的点积。</returns>
        public real_t Dot(Vector2 with)
        {
            return (x * with.x) + (y * with.y);
        }

        /// <summary>
        /// 返回一个所有分量向下舍入（向负无穷大）的新向量。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Floor"/> 的向量。</returns>
        public Vector2 Floor()
        {
            return new Vector2(Mathf.Floor(x), Mathf.Floor(y));
        }

        /// <summary>
        /// 返回此向量的倒数。 这与 <c>new Vector2(1 / v.x, 1 / v.y)</c> 相同。
        /// </summary>
        /// <returns>这个向量的倒数。</returns>
        public Vector2 Inverse()
        {
            return new Vector2(1 / x, 1 / y);
        }

        /// <summary>
        /// 如果向量被规范化，则返回 <see langword="true"/>，否则返回 <see langword="false"/>。
        /// </summary>
        /// <returns>一个 <see langword="bool"/> 表示向量是否被归一化。</returns>
        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared() - 1.0f) < Mathf.Epsilon;
        }

        /// <summary>
        /// 返回此向量的长度（大小）。
        /// </summary>
        /// <seealso cref="LengthSquared"/>
        /// <returns>这个向量的长度。</returns>
        public real_t Length()
        {
            return Mathf.Sqrt((x * x) + (y * y));
        }

        /// <summary>
        /// 返回此向量的平方长度（平方大小）。
        /// 这个方法比 <see cref="Length"/> 运行得快，所以如果
        /// 您需要比较向量或某些公式的平方长度。
        /// </summary>
        /// <returns>这个向量的平方长度。</returns>
        public real_t LengthSquared()
        {
            return (x * x) + (y * y);
        }

        /// <summary>
        /// 返回之间的线性插值结果
        /// 这个向量和 <paramref name="to"/> 的数量 <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">插值的目标向量。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果向量。</returns>
        public Vector2 LinearInterpolate(Vector2 to, real_t weight)
        {
            return new Vector2
            (
                Mathf.Lerp(x, to.x, weight),
                Mathf.Lerp(y, to.y, weight)
            );
        }

        /// <summary>
        /// 返回之间的线性插值结果
        /// 这个向量和 <paramref name="to"/> 的向量量 <paramref name="weight"/>。
        /// </summary>
        /// <param name="to">插值的目标向量.</param>
        /// <param name="weight">
        /// 分量在 0.0 到 1.0 范围内的向量，表示插值量。
        /// </param>
        /// <returns>插值的结果向量。</returns>
        public Vector2 LinearInterpolate(Vector2 to, Vector2 weight)
        {
            return new Vector2
            (
                Mathf.Lerp(x, to.x, weight.x),
                Mathf.Lerp(y, to.y, weight.y)
            );
        }

        /// <summary>
        /// Returns the vector with a maximum length by limiting its length to <paramref name="length"/>.
        /// </summary>
        /// <param name="length">The length to limit to.</param>
        /// <returns>The vector with its length limited.</returns>
        public Vector2 LimitLength(real_t length = 1.0f)
        {
            Vector2 v = this;
            real_t l = Length();

            if (l > 0 && length < l)
            {
                v /= l;
                v *= length;
            }

            return v;
        }

        /// <summary>
        /// Returns the axis of the vector's largest value. See <see cref="Axis"/>.
        /// If both components are equal, this method returns <see cref="Axis.X"/>.
        /// 返回向量最大值的轴。 请参阅 <see cref="Axis"/>。
        /// 如果两个分量相等，则此方法返回 <see cref="Axis.X"/>。

        /// </summary>
        /// <returns>最大轴的索引。</returns>
        public Axis MaxAxis()
        {
            return x < y ? Axis.Y : Axis.X;
        }

        /// <summary>
        /// 返回向量最小值的轴。 请参阅 <see cref="Axis"/>。
        /// 如果两个分量相等，则此方法返回 <see cref="Axis.Y"/>。
        /// </summary>
        /// <returns>最小轴的索引。</returns>
        public Axis MinAxis()
        {
            return x < y ? Axis.X : Axis.Y;
        }

        /// <summary>
        /// 将此向量向 <paramref name="to"/> 移动固定的 <paramref name="delta"/> 数量。
        /// </summary>
        /// <param name="to">要移动的向量。</param>
        /// <param name="delta">要移动的数量。</param>
        /// <returns>结果向量。</returns>
        public Vector2 MoveToward(Vector2 to, real_t delta)
        {
            Vector2 v = this;
            Vector2 vd = to - v;
            real_t len = vd.Length();
            if (len <= delta || len < Mathf.Epsilon)
                return to;

            return v + (vd / len * delta);
        }

        /// <summary>
        /// 返回缩放到单位长度的向量。 等效于 <c>v / v.Length()</c>。
        /// </summary>
        /// <returns>向量的标准化版本。</returns>
        public Vector2 Normalized()
        {
            Vector2 v = this;
            v.Normalize();
            return v;
        }

        /// <summary>
        /// 返回一个逆时针旋转 90 度的垂直向量
        /// 与原始相比，长度相同。
        /// </summary>
        /// <returns>垂直向量。</returns>
        public Vector2 Perpendicular()
        {
            return new Vector2(y, -x);
        }

        /// <summary>
        /// 返回一个由该向量的组件的 <see cref="Mathf.PosMod(real_t, real_t)"/> 组成的向量
        /// 和 <paramref name="mod"/>。
        /// </summary>
        /// <param name="mod">表示运算除数的值。</param>
        /// <returns>
        /// 每个分量的向量 <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="mod"/>。
        /// </returns>
        public Vector2 PosMod(real_t mod)
        {
            Vector2 v;
            v.x = Mathf.PosMod(x, mod);
            v.y = Mathf.PosMod(y, mod);
            return v;
        }

        /// <summary>
        /// 返回一个由该向量的组件的 <see cref="Mathf.PosMod(real_t, real_t)"/> 组成的向量
        /// 和 <paramref name="modv"/> 的组件。
        /// </summary>
        /// <param name="modv">表示运算除数的向量。</param>
        /// <returns>
        /// 每个组件的向量 <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="modv"/> 的组件。
        /// </returns>
        public Vector2 PosMod(Vector2 modv)
        {
            Vector2 v;
            v.x = Mathf.PosMod(x, modv.x);
            v.y = Mathf.PosMod(y, modv.y);
            return v;
        }

        /// <summary>
        /// 返回这个向量投影到另一个向量 <paramref name="onNormal"/>。
        /// </summary>
        /// <param name="onNormal">要投影到的向量。</param>
        /// <returns>投影向量。</returns>
        public Vector2 Project(Vector2 onNormal)
        {
            return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
        }

        /// <summary>
        /// 返回从给定 <paramref name="normal"/> 定义的平面反射的向量。
        /// </summary>
        /// <param name="normal">定义要反射的平面的法线向量。 必须标准化。</param>
        /// <returns>反射向量。</returns>
        public Vector2 Reflect(Vector2 normal)
        {
#if DEBUG
            if (!normal.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(normal));
            }
#endif
            return (2 * Dot(normal) * normal) - this;
        }

        /// <summary>
        /// 将此向量旋转 <paramref name="phi"/> 弧度。
        /// </summary>
        /// <param name="phi">旋转角度，以弧度为单位。</param>
        /// <returns>旋转后的向量。</returns>
        public Vector2 Rotated(real_t phi)
        {
            real_t sine = Mathf.Sin(phi);
            real_t cosi = Mathf.Cos(phi);
            return new Vector2(
                x * cosi - y * sine,
                x * sine + y * cosi);
        }

        /// <summary>
        /// 返回这个向量，所有分量都四舍五入到最接近的整数，
        /// 中间的情况向最接近的二的倍数舍入。
        /// </summary>
        /// <returns>圆角向量。</returns>
        public Vector2 Round()
        {
            return new Vector2(Mathf.Round(x), Mathf.Round(y));
        }

        [Obsolete("Set is deprecated. Use the Vector2(" + nameof(real_t) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void Set(real_t x, real_t y)
        {
            this.x = x;
            this.y = y;
        }
        [Obsolete("Set is deprecated. Use the Vector2(" + nameof(Vector2) + ") constructor instead.", error: true)]
        public void Set(Vector2 v)
        {
            x = v.x;
            y = v.y;
        }

        /// <summary>
        /// 返回一个向量，每个分量设置为一或负一，具体取决于
        /// 在这个向量的分量的符号上，如果分量为零，则为零，
        /// 通过在每个组件上调用 <see cref="Mathf.Sign(real_t)"/>。
        /// </summary>
        /// <returns>一个向量，其所有分量为 <c>1</c>、<c>-1</c> 或 <c>0</c>。</returns>
        public Vector2 Sign()
        {
            Vector2 v;
            v.x = Mathf.Sign(x);
            v.y = Mathf.Sign(y);
            return v;
        }

        /// <summary>
        /// 返回球面线性插值的结果
        /// 这个向量和 <paramref name="to"/> 的数量 <paramref name="weight"/>.
        ///
        /// 注意：两个向量都必须归一化。
        /// </summary>
        /// <param name="to">插值的目标向量。 必须标准化。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果向量。</returns>
        public Vector2 Slerp(Vector2 to, real_t weight)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Vector2.Slerp: From vector is not normalized.");
            }
            if (!to.IsNormalized())
            {
                throw new InvalidOperationException($"Vector2.Slerp: `{nameof(to)}` is not normalized.");
            }
#endif
            return Rotated(AngleTo(to) * weight);
        }

        /// <summary>
        /// 返回这个向量沿着给定的 <paramref name="normal"/> 定义的平面滑动。
        /// </summary>
        /// <param name="normal">定义要在其上滑动的平面的法线向量。</param>
        /// <returns>滑动向量。</returns>
        public Vector2 Slide(Vector2 normal)
        {
            return this - (normal * Dot(normal));
        }

        /// <summary>
        /// 返回此向量，其中每个组件都捕捉到最接近的 <paramref name="step"/> 倍数。
        /// 这也可以用于四舍五入到任意小数位数。
        /// </summary>
        /// <param name="step">一个向量值，表示要捕捉到的步长。</param>
        /// <returns>捕捉到的向量。</returns>
        public Vector2 Snapped(Vector2 step)
        {
            return new Vector2(Mathf.Stepify(x, step.x), Mathf.Stepify(y, step.y));
        }

        /// <summary>
        /// 返回一个逆时针旋转 90 度的垂直向量
        /// 与原始相比，长度相同。
        /// 已弃用，将在 4.0 中替换为 <see cref="Perpendicular"/>。
        /// </summary>
        /// <returns>垂直向量。</returns>
        public Vector2 Tangent()
        {
            return new Vector2(y, -x);
        }

        // Constants
        private static readonly Vector2 _zero = new Vector2(0, 0);
        private static readonly Vector2 _one = new Vector2(1, 1);
        private static readonly Vector2 _negOne = new Vector2(-1, -1);
        private static readonly Vector2 _inf = new Vector2(Mathf.Inf, Mathf.Inf);

        private static readonly Vector2 _up = new Vector2(0, -1);
        private static readonly Vector2 _down = new Vector2(0, 1);
        private static readonly Vector2 _right = new Vector2(1, 0);
        private static readonly Vector2 _left = new Vector2(-1, 0);

        /// <summary>
        /// 零向量，所有分量都设置为<c>0</c>的向量。
        /// </summary>
        /// <value>等价于<c>new Vector2(0, 0)</c>.</value>
        public static Vector2 Zero { get { return _zero; } }
        /// <summary>
        /// 已弃用，请在 <see cref="One"/> 中使用负号。
        /// </summary>
        /// <value>等价于<c>new Vector2(-1, -1)</c>.</value>
        [Obsolete("Use a negative sign with Vector2.One instead.")]
        public static Vector2 NegOne { get { return _negOne; } }
        /// <summary>
        /// 一个向量，一个所有分量都设置为<c>1</c>的向量。
        /// </summary>
        /// <value>等价于<c>new Vector2(1, 1)</c>.</value>
        public static Vector2 One { get { return _one; } }
        /// <summary>
        /// 无穷大向量，所有分量都设置为 <see cref="Mathf.Inf"/> 的向量。
        /// </summary>
        /// <value>等价于<c>new Vector2(Mathf.Inf, Mathf.Inf)</c>.</value>
        public static Vector2 Inf { get { return _inf; } }

        /// <summary>
        /// 向上单位向量。 Y 在 2D 中向下，所以这个向量指向 -Y。
        /// </summary>
        /// <value>等价于<c>new Vector2(0, -1)</c>.</value>
        public static Vector2 Up { get { return _up; } }
        /// <summary>
        /// 向下单位向量。 Y 在 2D 中向下，所以这个向量指向 +Y。
        /// </summary>
        /// <value>等价于<c>new Vector2(0, 1)</c>.</value>
        public static Vector2 Down { get { return _down; } }
        /// <summary>
        /// 右单位向量。 代表正确的方向。
        /// </summary>
        /// <value>等价于<c>new Vector2(1, 0)</c>.</value>
        public static Vector2 Right { get { return _right; } }
        /// <summary>
        /// 左单位向量。 代表左的方向。
        /// </summary>
        /// <value>等价于<c>new Vector2(-1, 0)</c>.</value>
        public static Vector2 Left { get { return _left; } }

        /// <summary>
        /// 用给定的组件构造一个新的 <see cref="Vector2"/>。
        /// </summary>
        /// <param name="x">向量的X分量。</param>
        /// <param name="y">向量的Y分量。</param>
        public Vector2(real_t x, real_t y)
        {
            this.x = x;
            this.y = y;
        }

        /// <summary>
        /// 从现有的 <see cref="Vector2"/> 构造一个新的 <see cref="Vector2"/>。
        /// </summary>
        /// <param name="v">现有的<see cref="Vector2"/>.</param>
        public Vector2(Vector2 v)
        {
            x = v.x;
            y = v.y;
        }

        /// <summary>
        /// Adds each component of the <see cref="Vector2"/>
        /// with the components of the given <see cref="Vector2"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector2 operator +(Vector2 left, Vector2 right)
        {
            left.x += right.x;
            left.y += right.y;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector2"/>
        /// by the components of the given <see cref="Vector2"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector2 operator -(Vector2 left, Vector2 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector2"/>.
        /// This is the same as writing <c>new Vector2(-v.x, -v.y)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// With floats, the number zero can be either positive or negative.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector2 operator -(Vector2 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector2"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector2 operator *(Vector2 vec, real_t scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector2"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector2 operator *(real_t scale, Vector2 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector2"/>
        /// by the components of the given <see cref="Vector2"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector2 operator *(Vector2 left, Vector2 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            return left;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector2"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector2 operator /(Vector2 vec, real_t divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector2"/>
        /// by the components of the given <see cref="Vector2"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector2 operator /(Vector2 vec, Vector2 divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector2"/>
        /// with the components of the given <see cref="real_t"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(real_t)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector2(10, -20) % 7); // Prints "(3, -6)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector2 operator %(Vector2 vec, real_t divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector2"/>
        /// with the components of the given <see cref="Vector2"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(Vector2)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector2(10, -20) % new Vector2(7, 8)); // Prints "(3, -4)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector2 operator %(Vector2 vec, Vector2 divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            return vec;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are exactly equal.</returns>
        public static bool operator ==(Vector2 left, Vector2 right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the vectors are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the vectors are not equal.</returns>
        public static bool operator !=(Vector2 left, Vector2 right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector2"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector2 left, Vector2 right)
        {
            if (left.x == right.x)
            {
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector2"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector2 left, Vector2 right)
        {
            if (left.x == right.x)
            {
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector2"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector2 left, Vector2 right)
        {
            if (left.x == right.x)
            {
                return left.y <= right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector2"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector2 left, Vector2 right)
        {
            if (left.x == right.x)
            {
                return left.y >= right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// 如果此向量和 <paramref name="obj"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="obj">要比较的另一个对象。</param>
        /// <returns>向量和其他对象是否相等。</returns>
        public override bool Equals(object obj)
        {
            if (obj is Vector2)
            {
                return Equals((Vector2)obj);
            }
            return false;
        }

        /// <summary>
        /// 如果此向量和 <paramref name="other"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="other">要比较的另一个向量。</param>
        /// <returns>向量是否相等。</returns>
        public bool Equals(Vector2 other)
        {
            return x == other.x && y == other.y;
        }

        /// <summary>
        /// 如果此向量和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Mathf.IsEqualApprox(real_t, real_t)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个向量。</param>
        /// <returns>向量是否近似相等。</returns>
        public bool IsEqualApprox(Vector2 other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y);
        }

        /// <summary>
        /// 用作 <see cref="Vector2"/> 的散列函数。
        /// </summary>
        /// <returns>这个向量的哈希码。</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Vector2"/> 转换为字符串。
        /// </summary>
        /// <returns>此向量的字符串表示形式。</returns>
        public override string ToString()
        {
            return $"({x}, {y})";
        }

        /// <summary>
        /// 将此 <see cref="Vector2"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此向量的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)})";
        }
    }
}
