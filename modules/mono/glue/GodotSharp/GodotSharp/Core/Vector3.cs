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
    /// 三元素结构，可用于表示 3D 空间中的位置或任何其他数值对。
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector3 : IEquatable<Vector3>
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
            Y,
            /// <summary>
            /// 向量的 Z 轴。
            /// </summary>
            Z
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
        /// 向量的 Z 分量。 也可以通过使用索引位置 <c>[2]</c> 访问。
        /// </summary>
        public real_t z;

        /// <summary>
        /// 使用它们的索引访问向量分量。
        /// </summary>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when the given the <paramref name="index"/> is not 0, 1 or 2.
        /// </exception>
        /// <value>
        /// <c>[0]</c> is equivalent to <see cref="x"/>,
        /// <c>[1]</c> is equivalent to <see cref="y"/>,
        /// <c>[2]</c> is equivalent to <see cref="z"/>.
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
                    case 2:
                        return z;
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
                    case 2:
                        z = value;
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
                x = y = z = 0f;
            }
            else
            {
                real_t length = Mathf.Sqrt(lengthsq);
                x /= length;
                y /= length;
                z /= length;
            }
        }

        /// <summary>
        /// 返回一个新向量，其中所有分量均为绝对值（即正数）。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Abs(real_t)"/> 的向量。</returns>
        public Vector3 Abs()
        {
            return new Vector3(Mathf.Abs(x), Mathf.Abs(y), Mathf.Abs(z));
        }

        /// <summary>
        /// 返回给定向量的无符号最小角度，以弧度为单位。
        /// </summary>
        /// <param name="to">与此向量进行比较的另一个向量。</param>
        /// <returns>两个向量之间的无符号角度，以弧度为单位。</returns>
        public real_t AngleTo(Vector3 to)
        {
            return Mathf.Atan2(Cross(to).Length(), Dot(to));
        }

        /// <summary>
        /// 从给定法线定义的平面返回此向量“反弹”。
        /// </summary>
        /// <param name="normal">定义要反弹的平面的法线向量。 必须标准化。</param>
        /// <returns>反弹的向量。</returns>
        public Vector3 Bounce(Vector3 normal)
        {
            return -Reflect(normal);
        }

        /// <summary>
        /// 返回一个新向量，其中所有分量向上舍入（朝向正无穷大）。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Ceil"/> 的向量。</returns>
        public Vector3 Ceil()
        {
            return new Vector3(Mathf.Ceil(x), Mathf.Ceil(y), Mathf.Ceil(z));
        }

        /// <summary>
        /// 返回此向量与 <paramref name="b"/> 的叉积。
        /// </summary>
        /// <param name="b">另一个向量。</param>
        /// <returns>叉积向量。</returns>
        public Vector3 Cross(Vector3 b)
        {
            return new Vector3
            (
                (y * b.z) - (z * b.y),
                (z * b.x) - (x * b.z),
                (x * b.y) - (y * b.x)
            );
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
        public Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, real_t weight)
        {
            Vector3 p0 = preA;
            Vector3 p1 = this;
            Vector3 p2 = b;
            Vector3 p3 = postB;

            real_t t = weight;
            real_t t2 = t * t;
            real_t t3 = t2 * t;

            return 0.5f * (
                (p1 * 2.0f) + ((-p0 + p2) * t) +
                (((2.0f * p0) - (5.0f * p1) + (4f * p2) - p3) * t2) +
                ((-p0 + (3.0f * p1) - (3.0f * p2) + p3) * t3)
            );
        }

        /// <summary>
        /// 返回从这个向量指向 <paramref name="b"/> 的归一化向量。
        /// </summary>
        /// <param name="b">另一个指向的向量。</param>
        /// <returns>从这个向量到<paramref name="b"/>的方向。</returns>
        public Vector3 DirectionTo(Vector3 b)
        {
            return new Vector3(b.x - x, b.y - y, b.z - z).Normalized();
        }

        /// <summary>
        /// 返回此向量与 <paramref name="b"/> 之间的平方距离。
        /// 这个方法比 <see cref="DistanceTo"/> 运行得快，所以如果
        /// 你需要比较向量或者需要一些公式的平方距离。
        /// </summary>
        /// <param name="b">要使用的另一个向量。</param>
        /// <returns>两个向量之间的平方距离。</returns>
        public real_t DistanceSquaredTo(Vector3 b)
        {
            return (b - this).LengthSquared();
        }

        /// <summary>
        /// 返回此向量与 <paramref name="b"/> 之间的距离。
        /// </summary>
        /// <seealso cref="DistanceSquaredTo(Vector3)"/>
        /// <param name="b">要使用的另一个向量。</param>
        /// <returns>两个向量之间的距离。</returns>
        public real_t DistanceTo(Vector3 b)
        {
            return (b - this).Length();
        }

        /// <summary>
        /// 返回此向量与 <paramref name="b"/> 的点积。
        /// </summary>
        /// <param name="b">要使用的另一个向量。</param>
        /// <returns>两个向量的点积。</returns>
        public real_t Dot(Vector3 b)
        {
            return (x * b.x) + (y * b.y) + (z * b.z);
        }

        /// <summary>
        /// 返回一个所有分量向下舍入（向负无穷大）的新向量。
        /// </summary>
        /// <returns>在每个组件上调用带有 <see cref="Mathf.Floor"/> 的向量。</returns>
        public Vector3 Floor()
        {
            return new Vector3(Mathf.Floor(x), Mathf.Floor(y), Mathf.Floor(z));
        }

        /// <summary>
        /// 返回此向量的逆。 这与 <c>new Vector3(1 / v.x, 1 / v.y, 1 / v.z)</c> 相同。
        /// </summary>
        /// <returns>这个向量的倒数。</returns>
        public Vector3 Inverse()
        {
            return new Vector3(1 / x, 1 / y, 1 / z);
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
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;

            return Mathf.Sqrt(x2 + y2 + z2);
        }

        /// <summary>
        /// 返回此向量的平方长度（平方大小）。
        /// 这个方法比 <see cref="Length"/> 运行得快，所以如果
        /// 您需要比较向量或某些公式的平方长度。
        /// </summary>
        /// <returns>这个向量的平方长度。</returns>
        public real_t LengthSquared()
        {
            real_t x2 = x * x;
            real_t y2 = y * y;
            real_t z2 = z * z;

            return x2 + y2 + z2;
        }

        /// <summary>
        /// 返回之间的线性插值结果
        /// 这个向量和 <paramref name="to"/> 的数量 <paramref name="weight"/>.
        /// </summary>
        /// <param name="to">插值的目标向量。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果向量。</returns>
        public Vector3 LinearInterpolate(Vector3 to, real_t weight)
        {
            return new Vector3
            (
                Mathf.Lerp(x, to.x, weight),
                Mathf.Lerp(y, to.y, weight),
                Mathf.Lerp(z, to.z, weight)
            );
        }

        /// <summary>
        /// 返回之间的线性插值结果
        /// 这个向量和 <paramref name="to"/> 的向量量 <paramref name="weight"/>。
        /// </summary>
        /// <param name="to">插值的目标向量。</param>
        /// <param name="weight">分量在0.0到1.0之间的向量，代表插值量。</param>
        /// <returns>插值的结果向量。</returns>
        public Vector3 LinearInterpolate(Vector3 to, Vector3 weight)
        {
            return new Vector3
            (
                Mathf.Lerp(x, to.x, weight.x),
                Mathf.Lerp(y, to.y, weight.y),
                Mathf.Lerp(z, to.z, weight.z)
            );
        }

        /// <summary>
        /// Returns the vector with a maximum length by limiting its length to <paramref name="length"/>.
        /// </summary>
        /// <param name="length">The length to limit to.</param>
        /// <returns>The vector with its length limited.</returns>
        public Vector3 LimitLength(real_t length = 1.0f)
        {
            Vector3 v = this;
            real_t l = Length();

            if (l > 0 && length < l)
            {
                v /= l;
                v *= length;
            }

            return v;
        }

        /// <summary>
        /// 返回向量最大值的轴。 请参阅 <see cref="Axis"/>。
        /// 如果所有组件都相等，则此方法返回 <see cref="Axis.X"/>。
        /// </summary>
        /// <returns>最大轴的索引。</returns>
        public Axis MaxAxis()
        {
            return x < y ? (y < z ? Axis.Z : Axis.Y) : (x < z ? Axis.Z : Axis.X);
        }

        /// <summary>
        /// 返回向量最小值的轴。 请参阅 <see cref="Axis"/>。
        /// 如果所有组件都相等，则此方法返回 <see cref="Axis.Z"/>。
        /// </summary>
        /// <returns>最小轴的索引。</returns>
        public Axis MinAxis()
        {
            return x < y ? (x < z ? Axis.X : Axis.Z) : (y < z ? Axis.Y : Axis.Z);
        }

        /// <summary>
        /// 将此向量向 <paramref name="to"/> 移动固定的 <paramref name="delta"/> 数量。
        /// </summary>
        /// <param name="to">要移动的向量。</param>
        /// <param name="delta">要移动的数量。</param>
        /// <returns>结果向量。</returns>
        public Vector3 MoveToward(Vector3 to, real_t delta)
        {
            Vector3 v = this;
            Vector3 vd = to - v;
            real_t len = vd.Length();
            if (len <= delta || len < Mathf.Epsilon)
                return to;

            return v + (vd / len * delta);
        }

        /// <summary>
        /// 返回缩放到单位长度的向量。 等效于 <c>v / v.Length()</c>。
        /// </summary>
        /// <returns>向量的标准化版本。</returns>
        public Vector3 Normalized()
        {
            Vector3 v = this;
            v.Normalize();
            return v;
        }

        /// <summary>
        /// 返回带有 <paramref name="b"/> 的外积。
        /// </summary>
        /// <param name="b">另一个向量。</param>
        /// <returns>代表外积矩阵的<see cref="Basis"/>。</returns>
        public Basis Outer(Vector3 b)
        {
            return new Basis(
                x * b.x, x * b.y, x * b.z,
                y * b.x, y * b.y, y * b.z,
                z * b.x, z * b.y, z * b.z
            );
        }

        /// <summary>
        /// 返回一个由该向量的组件的 <see cref="Mathf.PosMod(real_t, real_t)"/> 组成的向量
        /// 和 <paramref name="mod"/>。
        /// </summary>
        /// <param name="mod">表示运算除数的值。</param>
        /// <returns>
        /// 每个分量的向量 <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="mod"/>。
        /// </returns>
        public Vector3 PosMod(real_t mod)
        {
            Vector3 v;
            v.x = Mathf.PosMod(x, mod);
            v.y = Mathf.PosMod(y, mod);
            v.z = Mathf.PosMod(z, mod);
            return v;
        }

        /// <summary>
        /// 返回一个由该向量的组件的 <see cref="Mathf.PosMod(real_t, real_t)"/> 组成的向量
        /// 和 <paramref name="modv"/> 的组件。
        /// </summary>
        /// <param name="modv">表示运算除数的向量。</param>
        /// <返回>
        /// 每个组件的向量 <see cref="Mathf.PosMod(real_t, real_t)"/> by <paramref name="modv"/> 的组件。
        /// </returns>
        public Vector3 PosMod(Vector3 modv)
        {
            Vector3 v;
            v.x = Mathf.PosMod(x, modv.x);
            v.y = Mathf.PosMod(y, modv.y);
            v.z = Mathf.PosMod(z, modv.z);
            return v;
        }

        /// <summary>
        /// 返回这个向量投影到另一个向量 <paramref name="onNormal"/>。
        /// </summary>
        /// <param name="onNormal">要投影到的向量。</param>
        /// <returns>投影向量。</returns>
        public Vector3 Project(Vector3 onNormal)
        {
            return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
        }

        /// <summary>
        /// 返回从给定 <paramref name="normal"/> 定义的平面反射的向量。
        /// </summary>
        /// <param name="normal">定义要反射的平面的法线向量。 必须标准化。</param>
        /// <returns>反射向量。</returns>
        public Vector3 Reflect(Vector3 normal)
        {
#if DEBUG
            if (!normal.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(normal));
            }
#endif
            return (2.0f * Dot(normal) * normal) - this;
        }

        /// <summary>
        /// 围绕给定的 <paramref name="axis"/> 向量旋转该向量 <paramref name="phi"/> 弧度。
        /// <paramref name="axis"/> 向量必须是归一化向量。
        /// </summary>
        /// <param name="axis">要旋转的向量。 必须标准化。</param>
        /// <param name="phi">旋转角度，以弧度为单位。</param>
        /// <returns>旋转后的向量。</returns>
        public Vector3 Rotated(Vector3 axis, real_t phi)
        {
#if DEBUG
            if (!axis.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(axis));
            }
#endif
            return new Basis(axis, phi).Xform(this);
        }

        /// <summary>
        /// 返回这个向量，所有分量都四舍五入到最接近的整数，
        /// 中间的情况向最接近的二的倍数舍入。
        /// </summary>
        /// <returns>圆角向量。</returns>
        public Vector3 Round()
        {
            return new Vector3(Mathf.Round(x), Mathf.Round(y), Mathf.Round(z));
        }

        [Obsolete("Set is deprecated. Use the Vector3(" + nameof(real_t) + ", " + nameof(real_t) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void Set(real_t x, real_t y, real_t z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        [Obsolete("Set is deprecated. Use the Vector3(" + nameof(Vector3) + ") constructor instead.", error: true)]
        public void Set(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }

        /// <summary>
        /// 返回一个向量，每个分量设置为一或负一，具体取决于
        /// 在这个向量的分量的符号上，如果分量为零，则为零，
        /// 通过在每个组件上调用 <see cref="Mathf.Sign(real_t)"/>。
        /// </summary>
        /// <returns>一个向量，其所有分量为 <c>1</c>、<c>-1</c> 或 <c>0</c>。</returns>
        public Vector3 Sign()
        {
            Vector3 v;
            v.x = Mathf.Sign(x);
            v.y = Mathf.Sign(y);
            v.z = Mathf.Sign(z);
            return v;
        }

        /// <summary>
        /// 返回给定向量的带符号角度，以弧度为单位。
        /// 逆时针方向的角度符号为正
        /// 查看时顺时针方向和负数
        /// 从 <paramref name="axis"/> 指定的一侧开始。
        /// </summary>
        /// <param name="to">与此向量进行比较的另一个向量。</param>
        /// <param name="axis">用于角度符号的参考轴。</param>
        /// <returns>两个向量之间的有符号角度，以弧度为单位。</returns>
        public real_t SignedAngleTo(Vector3 to, Vector3 axis)
        {
            Vector3 crossTo = Cross(to);
            real_t unsignedAngle = Mathf.Atan2(crossTo.Length(), Dot(to));
            real_t sign = crossTo.Dot(axis);
            return (sign < 0) ? -unsignedAngle : unsignedAngle;
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
        public Vector3 Slerp(Vector3 to, real_t weight)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Vector3.Slerp: From vector is not normalized.");
            }
            if (!to.IsNormalized())
            {
                throw new InvalidOperationException($"Vector3.Slerp: `{nameof(to)}` is not normalized.");
            }
#endif
            real_t theta = AngleTo(to);
            return Rotated(Cross(to), theta * weight);
        }

        /// <summary>
        /// 返回这个向量沿着给定的 <paramref name="normal"/> 定义的平面滑动。
        /// </summary>
        /// <param name="normal">定义要在其上滑动的平面的法线向量。</param>
        /// <returns>滑动向量。</returns>
        public Vector3 Slide(Vector3 normal)
        {
            return this - (normal * Dot(normal));
        }

        /// <summary>
        /// 返回此向量，其中每个组件都捕捉到最接近的 <paramref name="step"/> 倍数。
        /// 这也可以用于四舍五入到任意小数位数。
        /// </summary>
        /// <param name="step">一个向量值，表示要捕捉到的步长。</param>
        /// <returns>捕捉到的向量。</returns>
        public Vector3 Snapped(Vector3 step)
        {
            return new Vector3
            (
                Mathf.Stepify(x, step.x),
                Mathf.Stepify(y, step.y),
                Mathf.Stepify(z, step.z)
            );
        }

        /// <summary>
        /// 返回一个以向量为主对角线的对角矩阵。
        ///
        /// 这相当于一个没有旋转或剪切的 <see cref="Basis"/> 和
        /// 这个向量的组件设置为比例。
        /// </summary>
        /// <returns>以向量为主对角线的<see cref="Basis"/>。</returns>
        public Basis ToDiagonalMatrix()
        {
            return new Basis(
                x, 0, 0,
                0, y, 0,
                0, 0, z
            );
        }

        // Constants
        private static readonly Vector3 _zero = new Vector3(0, 0, 0);
        private static readonly Vector3 _one = new Vector3(1, 1, 1);
        private static readonly Vector3 _negOne = new Vector3(-1, -1, -1);
        private static readonly Vector3 _inf = new Vector3(Mathf.Inf, Mathf.Inf, Mathf.Inf);

        private static readonly Vector3 _up = new Vector3(0, 1, 0);
        private static readonly Vector3 _down = new Vector3(0, -1, 0);
        private static readonly Vector3 _right = new Vector3(1, 0, 0);
        private static readonly Vector3 _left = new Vector3(-1, 0, 0);
        private static readonly Vector3 _forward = new Vector3(0, 0, -1);
        private static readonly Vector3 _back = new Vector3(0, 0, 1);

        /// <summary>
        /// 零向量，所有分量都设置为 <c>0</c> 的向量。
        /// </summary>
        /// <value>等价于<c>new Vector3(0, 0, 0)</c>。</value>
        public static Vector3 Zero { get { return _zero; } }
        /// <summary>
        /// 一个向量，所有分量都设置为 <c>1</c> 的向量。
        /// </summary>
        /// <value>等价于<c>new Vector3(1, 1, 1)</c>。</value>
        public static Vector3 One { get { return _one; } }
        /// <summary>
        /// 已弃用，请改用带有 <see cref="One"/> 的负号。
        /// </summary>
        /// <value>等价于<c>new Vector3(-1, -1, -1)</c>。</value>
        [Obsolete("Use a negative sign with Vector3.One instead.")]
        public static Vector3 NegOne { get { return _negOne; } }
        /// <summary>
        /// 无穷向量，所有分量都设置为 <see cref="Mathf.Inf"/> 的向量。
        /// </summary>
        /// <value>等价于<c>new Vector3(Mathf.Inf, Mathf.Inf, Mathf.Inf)</c>。</value>
        public static Vector3 Inf { get { return _inf; } }

        /// <summary>
        /// 向上单位向量。
        /// </summary>
        /// <value>等价于<c>new Vector3(0, 1, 0)</c>。</value>
        public static Vector3 Up { get { return _up; } }
        /// <summary>
        /// 向下单位向量。
        /// </summary>
        /// <value>等价于<c>new Vector3(0, -1, 0)</c>。</value>
        public static Vector3 Down { get { return _down; } }
        /// <summary>
        /// 右单位向量。 代表右的局部方向，
        /// 和东的全球方向。
        /// </summary>
        /// <value>等价于<c>new Vector3(1, 0, 0)</c>。</value>
        public static Vector3 Right { get { return _right; } }
        /// <summary>
        /// 左单位向量。 代表左的局部方向，
        /// 和西方的全球方向。
        /// </summary>
        /// <value>等价于 <c>new Vector3(-1, 0, 0)</c>.</value>。
        public static Vector3 Left { get { return _left; } }
        /// <summary>
        /// 前向单位向量。 代表本地前进方向，
        /// 和北的全球方向。
        /// </summary>
        /// <value>等价于<c>new Vector3(0, 0, -1)</c>.</value>
        public static Vector3 Forward { get { return _forward; } }
        /// <summary>
        /// 后单位向量。 代表背部的局部方向，
        /// 和南的全球方向。
        /// </summary>
        /// <value>等价于<c>new Vector3(0, 0, 1)</c>.</value>
        public static Vector3 Back { get { return _back; } }

        /// <summary>
        /// 用给定的组件构造一个新的 <see cref="Vector3"/>。
        /// </summary>
        /// <param name="x">向量的X分量。</param>
        /// <param name="y">向量的Y分量。</param>
        /// <param name="z">向量的Z分量。</param>
        public Vector3(real_t x, real_t y, real_t z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        /// <summary>
        /// 从现有的 <see cref="Vector3"/> 构造一个新的 <see cref="Vector3"/>。
        /// </summary>
        /// <param name="v">现有的<see cref="Vector3"/>.</param>
        public Vector3(Vector3 v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }

        /// <summary>
        /// Adds each component of the <see cref="Vector3"/>
        /// with the components of the given <see cref="Vector3"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The added vector.</returns>
        public static Vector3 operator +(Vector3 left, Vector3 right)
        {
            left.x += right.x;
            left.y += right.y;
            left.z += right.z;
            return left;
        }

        /// <summary>
        /// Subtracts each component of the <see cref="Vector3"/>
        /// by the components of the given <see cref="Vector3"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The subtracted vector.</returns>
        public static Vector3 operator -(Vector3 left, Vector3 right)
        {
            left.x -= right.x;
            left.y -= right.y;
            left.z -= right.z;
            return left;
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Vector3"/>.
        /// This is the same as writing <c>new Vector3(-v.x, -v.y, -v.z)</c>.
        /// This operation flips the direction of the vector while
        /// keeping the same magnitude.
        /// With floats, the number zero can be either positive or negative.
        /// </summary>
        /// <param name="vec">The vector to negate/flip.</param>
        /// <returns>The negated/flipped vector.</returns>
        public static Vector3 operator -(Vector3 vec)
        {
            vec.x = -vec.x;
            vec.y = -vec.y;
            vec.z = -vec.z;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The vector to multiply.</param>
        /// <param name="scale">The scale to multiply by.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3 operator *(Vector3 vec, real_t scale)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="scale">The scale to multiply by.</param>
        /// <param name="vec">The vector to multiply.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3 operator *(real_t scale, Vector3 vec)
        {
            vec.x *= scale;
            vec.y *= scale;
            vec.z *= scale;
            return vec;
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Vector3"/>
        /// by the components of the given <see cref="Vector3"/>.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>The multiplied vector.</returns>
        public static Vector3 operator *(Vector3 left, Vector3 right)
        {
            left.x *= right.x;
            left.y *= right.y;
            left.z *= right.z;
            return left;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector3"/>
        /// by the given <see cref="real_t"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3 operator /(Vector3 vec, real_t divisor)
        {
            vec.x /= divisor;
            vec.y /= divisor;
            vec.z /= divisor;
            return vec;
        }

        /// <summary>
        /// Divides each component of the <see cref="Vector3"/>
        /// by the components of the given <see cref="Vector3"/>.
        /// </summary>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The divided vector.</returns>
        public static Vector3 operator /(Vector3 vec, Vector3 divisorv)
        {
            vec.x /= divisorv.x;
            vec.y /= divisorv.y;
            vec.z /= divisorv.z;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3"/>
        /// with the components of the given <see cref="real_t"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(real_t)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3(10, -20, 30) % 7); // Prints "(3, -6, 2)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisor">The divisor value.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3 operator %(Vector3 vec, real_t divisor)
        {
            vec.x %= divisor;
            vec.y %= divisor;
            vec.z %= divisor;
            return vec;
        }

        /// <summary>
        /// Gets the remainder of each component of the <see cref="Vector3"/>
        /// with the components of the given <see cref="Vector3"/>.
        /// This operation uses truncated division, which is often not desired
        /// as it does not work well with negative numbers.
        /// Consider using <see cref="PosMod(Vector3)"/> instead
        /// if you want to handle negative numbers.
        /// </summary>
        /// <example>
        /// <code>
        /// GD.Print(new Vector3(10, -20, 30) % new Vector3(7, 8, 9)); // Prints "(3, -4, 3)"
        /// </code>
        /// </example>
        /// <param name="vec">The dividend vector.</param>
        /// <param name="divisorv">The divisor vector.</param>
        /// <returns>The remainder vector.</returns>
        public static Vector3 operator %(Vector3 vec, Vector3 divisorv)
        {
            vec.x %= divisorv.x;
            vec.y %= divisorv.y;
            vec.z %= divisorv.z;
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
        public static bool operator ==(Vector3 left, Vector3 right)
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
        public static bool operator !=(Vector3 left, Vector3 right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Compares two <see cref="Vector3"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than the right.</returns>
        public static bool operator <(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z < right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than the right.</returns>
        public static bool operator >(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z > right.z;
                }
                return left.y > right.y;
            }
            return left.x > right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is less than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is less than or equal to the right.</returns>
        public static bool operator <=(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z <= right.z;
                }
                return left.y < right.y;
            }
            return left.x < right.x;
        }

        /// <summary>
        /// Compares two <see cref="Vector3"/> vectors by first checking if
        /// the X value of the <paramref name="left"/> vector is greater than
        /// or equal to the X value of the <paramref name="right"/> vector.
        /// If the X values are exactly equal, then it repeats this check
        /// with the Y values of the two vectors, and then with the Z values.
        /// This operator is useful for sorting vectors.
        /// </summary>
        /// <param name="left">The left vector.</param>
        /// <param name="right">The right vector.</param>
        /// <returns>Whether or not the left is greater than or equal to the right.</returns>
        public static bool operator >=(Vector3 left, Vector3 right)
        {
            if (left.x == right.x)
            {
                if (left.y == right.y)
                {
                    return left.z >= right.z;
                }
                return left.y > right.y;
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
            if (obj is Vector3)
            {
                return Equals((Vector3)obj);
            }

            return false;
        }

        /// <summary>
        /// 如果此向量和 <paramref name="other"/> 相等，则返回 <see langword="true"/>
        /// </summary>
        /// <param name="other">要比较的另一个向量。</param>
        /// <returns>向量是否相等。</returns>
        public bool Equals(Vector3 other)
        {
            return x == other.x && y == other.y && z == other.z;
        }

        /// <summary>
        /// 如果此向量和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Mathf.IsEqualApprox(real_t, real_t)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个向量。</param>
        /// <returns>向量是否近似相等。</returns>
        public bool IsEqualApprox(Vector3 other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z);
        }

        /// <summary>
        /// 用作 <see cref="Vector3"/> 的哈希函数。
        /// </summary>
        /// <returns>这个向量的哈希码。</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Vector3"/> 转换为字符串。
        /// </summary>
        /// <returns>此向量的字符串表示形式。</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z})";
        }

        /// <summary>
        /// 将此 <see cref="Vector3"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此向量的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)})";
        }
    }
}
