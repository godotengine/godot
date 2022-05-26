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
    /// 用于表示 3D 旋转的单位四元数。
    /// 四元数需要归一化才能用于旋转。
    ///
    /// 类似于<see cref="Basis"/>，实现矩阵
    /// 旋转的表示，并且可以使用两者进行参数化
    /// 一个轴角对或欧拉角。 基础存储旋转、缩放、
    /// 和剪切，而 Quat 只存储旋转。
    ///
    /// 由于它的紧凑性和它在内存中的存储方式，某些
    /// 操作（尤其是获取轴角和执行 SLERP）
    /// 对浮点错误更有效和更健壮。
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Quat : IEquatable<Quat>
    {
        /// <summary>
        /// 四元数的 X 分量（虚构的 <c>i</c> 轴部分）。
        /// 四元数组件通常不应该被直接操作。
        /// </summary>
        public real_t x;

        /// <summary>
        /// 四元数的Y分量（虚<c>j</c>轴部分）。
        /// 四元数组件通常不应该被直接操作。
        /// </summary>
        public real_t y;

        /// <summary>
        /// 四元数的 Z 分量（虚构的 <c>k</c> 轴部分）。
        /// 四元数组件通常不应该被直接操作。
        /// </summary>
        public real_t z;

        /// <summary>
        /// 四元数的 W 分量（实部）。
        /// 四元数组件通常不应该被直接操作。
        /// </summary>
        public real_t w;

        /// <summary>
        /// 使用它们的索引访问四元数组件。
        /// </summary>
        /// <value>
        /// <c>[0]</c> 等价于 <see cref="x"/>,
        /// <c>[1]</c> 等价于 <see cref="y"/>,
        /// <c>[2]</c> 等价于 <see cref="z"/>,
        /// <c>[3]</c> 等价于 <see cref="w"/>。
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
                    case 3:
                        return w;
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
                        break;
                    case 1:
                        y = value;
                        break;
                    case 2:
                        z = value;
                        break;
                    case 3:
                        w = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// 返回四元数的长度（大小）。
        /// </summary>
        /// <seealso cref="LengthSquared"/>
        /// <value>等价于<c>Mathf.Sqrt(LengthSquared)</c>.</value>
        public real_t Length
        {
            get { return Mathf.Sqrt(LengthSquared); }
        }

        /// <summary>
        /// 返回四元数的平方长度（平方大小）。
        /// 这个方法比 <see cref="Length"/> 运行得快，所以如果
        /// 您需要比较四元数或某些公式的平方长度。
        /// </summary>
        /// <value>等价于<c>Dot(this)</c>.</value>
        public real_t LengthSquared
        {
            get { return Dot(this); }
        }

        /// <summary>
        /// 返回此四元数与 <paramref name="to"/> 之间的角度。
        /// 这是您需要旋转的角度的大小
        /// 通过从一个到另一个。
        ///
        /// 注意：此方法金额异常高
        /// 的浮点错误，所以方法如
        /// <see cref="Mathf.IsZeroApprox"/> 将无法可靠地工作。
        /// </summary>
        /// <param name="to">另一个四元数。</param>
        /// <returns>四元数之间的角度。</returns>
        public real_t AngleTo(Quat to)
        {
            real_t dot = Dot(to);
            return Mathf.Acos(Mathf.Clamp(dot * dot * 2 - 1, -1, 1));
        }

        /// <summary>
        /// 在四元数之间执行三次球面插值 <paramref name="preA"/>，这个四元数，
        /// <paramref name="b"/> 和 <paramref name="postB"/>，按给定的数量 <paramref name="weight"/>。
        /// </summary>
        /// <param name="b">目标四元数。</param>
        /// <param name="preA">这个四元数之前的一个四元数。</param>
        /// <param name="postB"><paramref name="b"/>之后的四元数。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>内插四元数。</returns>
        public Quat CubicSlerp(Quat b, Quat preA, Quat postB, real_t weight)
        {
            real_t t2 = (1.0f - weight) * weight * 2f;
            Quat sp = Slerp(b, weight);
            Quat sq = preA.Slerpni(postB, weight);
            return sp.Slerpni(sq, t2);
        }

        /// <summary>
        /// 返回两个四元数的点积。
        /// </summary>
        /// <param name="b">另一个四元数。</param>
        /// <returns>点积。</returns>
        public real_t Dot(Quat b)
        {
            return (x * b.x) + (y * b.y) + (z * b.z) + (w * b.w);
        }

        /// <summary>
        /// 返回欧拉角（在 YXZ 约定中：分解时，
        /// 先是Z，然后是X，最后是Y）对应旋转
        /// 由单位四元数表示。 返回的向量包含
        /// 格式中的旋转角度（X 角度，Y 角度，Z 角度）。
        /// </summary>
        /// <returns>这个四元数的欧拉角表示。</returns>
        public Vector3 GetEuler()
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            var basis = new Basis(this);
            return basis.GetEuler();
        }

        /// <summary>
        /// 返回四元数的逆。
        /// </summary>
        /// <returns>反四元数。</returns>
        public Quat Inverse()
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            return new Quat(-x, -y, -z, w);
        }

        /// <summary>
        ///返回四元数是否标准化。
        /// </summary>
        /// <returns>一个 <see langword="bool"/> 四元数是否被规范化。</returns>
        public bool IsNormalized()
        {
            return Mathf.Abs(LengthSquared - 1) <= Mathf.Epsilon;
        }

        /// <summary>
        /// 返回四元数的副本，标准化为单位长度。
        /// </summary>
        /// <returns>标准化的四元数。</returns>
        public Quat Normalized()
        {
            return this / Length;
        }

        [Obsolete("Set is deprecated. Use the Quat(" + nameof(real_t) + ", " + nameof(real_t) + ", " + nameof(real_t) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void Set(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        [Obsolete("Set is deprecated. Use the Quat(" + nameof(Quat) + ") constructor instead.", error: true)]
        public void Set(Quat q)
        {
            this = q;
        }

        [Obsolete("SetAxisAngle is deprecated. Use the Quat(" + nameof(Vector3) + ", " + nameof(real_t) + ") constructor instead.", error: true)]
        public void SetAxisAngle(Vector3 axis, real_t angle)
        {
            this = new Quat(axis, angle);
        }

        [Obsolete("SetEuler is deprecated. Use the Quat(" + nameof(Vector3) + ") constructor instead.", error: true)]
        public void SetEuler(Vector3 eulerYXZ)
        {
            this = new Quat(eulerYXZ);
        }

        /// <summary>
        /// 返回球面线性插值的结果
        /// 这个四元数和 <paramref name="to"/> 的数量 <paramref name="weight"/>.
        ///
        /// 注意：两个四元数都必须标准化。
        /// </summary>
        /// <param name="to">插值的目标四元数。 必须标准化。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果四元数。</returns>
        public Quat Slerp(Quat to, real_t weight)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
            if (!to.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(to));
            }
#endif

            // Calculate cosine.
            real_t cosom = x * to.x + y * to.y + z * to.z + w * to.w;

            var to1 = new Quat();

            // Adjust signs if necessary.
            if (cosom < 0.0)
            {
                cosom = -cosom;
                to1.x = -to.x;
                to1.y = -to.y;
                to1.z = -to.z;
                to1.w = -to.w;
            }
            else
            {
                to1.x = to.x;
                to1.y = to.y;
                to1.z = to.z;
                to1.w = to.w;
            }

            real_t sinom, scale0, scale1;

            // Calculate coefficients.
            if (1.0 - cosom > Mathf.Epsilon)
            {
                // Standard case (Slerp).
                real_t omega = Mathf.Acos(cosom);
                sinom = Mathf.Sin(omega);
                scale0 = Mathf.Sin((1.0f - weight) * omega) / sinom;
                scale1 = Mathf.Sin(weight * omega) / sinom;
            }
            else
            {
                // Quaternions are very close so we can do a linear interpolation.
                scale0 = 1.0f - weight;
                scale1 = weight;
            }

            // Calculate final values.
            return new Quat
            (
                (scale0 * x) + (scale1 * to1.x),
                (scale0 * y) + (scale1 * to1.y),
                (scale0 * z) + (scale1 * to1.z),
                (scale0 * w) + (scale1 * to1.w)
            );
        }

        /// <summary>
        /// 返回球面线性插值的结果
        /// 这个四元数和 <paramref name="to"/> 按数量 <paramref name="weight"/>，但没有
        /// 检查旋转路径是否不大于 90 度。
        /// </summary>
        /// <param name="to">插值的目标四元数。 必须标准化。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值的结果四元数。</returns>
        public Quat Slerpni(Quat to, real_t weight)
        {
            real_t dot = Dot(to);

            if (Mathf.Abs(dot) > 0.9999f)
            {
                return this;
            }

            real_t theta = Mathf.Acos(dot);
            real_t sinT = 1.0f / Mathf.Sin(theta);
            real_t newFactor = Mathf.Sin(weight * theta) * sinT;
            real_t invFactor = Mathf.Sin((1.0f - weight) * theta) * sinT;

            return new Quat
            (
                (invFactor * x) + (newFactor * to.x),
                (invFactor * y) + (newFactor * to.y),
                (invFactor * z) + (newFactor * to.z),
                (invFactor * w) + (newFactor * to.w)
            );
        }

        /// <summary>
        /// 返回一个被这个四元数转换（相乘）的向量。
        /// </summary>
        /// <param name="v">要转换的向量。</param>
        /// <returns>转换后的向量。</returns>
        public Vector3 Xform(Vector3 v)
        {
#if DEBUG
            if (!IsNormalized())
            {
                throw new InvalidOperationException("Quat is not normalized");
            }
#endif
            var u = new Vector3(x, y, z);
            Vector3 uv = u.Cross(v);
            return v + (((uv * w) + u.Cross(uv)) * 2);
        }

        // Constants
        private static readonly Quat _identity = new Quat(0, 0, 0, 1);

        /// <summary>
        /// 恒等四元数，表示没有旋转。
        /// 等价于一个恒等<see cref="Basis"/> 矩阵。 如果一个向量被转换为
        /// 一个身份四元数，它不会改变。
        /// </summary>
        /// <value>等价于<c>new Quat(0, 0, 0, 1)</c>.</value>
        public static Quat Identity { get { return _identity; } }

        /// <summary>
        /// 构造一个由给定值定义的 <see cref="Quat"/>。
        /// </summary>
        /// <param name="x">四元数的X分量（虚<c>i</c>轴部分）。</param>
        /// <param name="y">四元数的Y分量（虚<c>j</c>轴部分）。</param>
        /// <param name="z">四元数的Z分量（虚<c>k</c>轴部分）。</param>
        /// <param name="w">四元数的W分量（实部）。</param>
        public Quat(real_t x, real_t y, real_t z, real_t w)
        {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        }

        /// <summary>
        /// 从给定的 <see cref="Quat"/> 构造一个 <see cref="Quat"/>。
        /// </summary>
        /// <param name="q">现有的四元数。</param>
        public Quat(Quat q)
        {
            this = q;
        }

        /// <summary>
        /// 从给定的 <see cref="Basis"/> 构造一个 <see cref="Quat"/>。
        /// </summary>
        /// <param name="basis">要构造的<see cref="Basis"/>。</param>
        public Quat(Basis basis)
        {
            this = basis.Quat();
        }

        /// <summary>
        /// 构造一个 <see cref="Quat"/> 将执行指定的旋转
        /// 欧拉角（在 YXZ 约定中：分解时，首先是 Z，然后是 X，最后是 Y），
        /// 以矢量格式给出（X 角，Y 角，Z 角）。
        /// </summary>
        /// <param name="eulerYXZ">四元数旋转的欧拉角。</param>
        public Quat(Vector3 eulerYXZ)
        {
            real_t halfA1 = eulerYXZ.y * 0.5f;
            real_t halfA2 = eulerYXZ.x * 0.5f;
            real_t halfA3 = eulerYXZ.z * 0.5f;

            // R = Y(a1).X(a2).Z(a3) convention for Euler angles.
            // Conversion to quaternion as listed in https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf (page A-6)
            // a3 is the angle of the first rotation, following the notation in this reference.

            real_t cosA1 = Mathf.Cos(halfA1);
            real_t sinA1 = Mathf.Sin(halfA1);
            real_t cosA2 = Mathf.Cos(halfA2);
            real_t sinA2 = Mathf.Sin(halfA2);
            real_t cosA3 = Mathf.Cos(halfA3);
            real_t sinA3 = Mathf.Sin(halfA3);

            x = (sinA1 * cosA2 * sinA3) + (cosA1 * sinA2 * cosA3);
            y = (sinA1 * cosA2 * cosA3) - (cosA1 * sinA2 * sinA3);
            z = (cosA1 * cosA2 * sinA3) - (sinA1 * sinA2 * cosA3);
            w = (sinA1 * sinA2 * sinA3) + (cosA1 * cosA2 * cosA3);
        }

        /// <summary>
        /// 构造一个将围绕给定轴旋转的 <see cref="Quat"/>
        /// 指定角度。 轴必须是归一化向量。
        /// </summary>
        /// <param name="axis">要旋转的轴。 必须标准化。</param>
        /// <param name="angle">要旋转的角度，以弧度为单位。</param>
        public Quat(Vector3 axis, real_t angle)
        {
#if DEBUG
            if (!axis.IsNormalized())
            {
                throw new ArgumentException("Argument is not normalized", nameof(axis));
            }
#endif

            real_t d = axis.Length();

            if (d == 0f)
            {
                x = 0f;
                y = 0f;
                z = 0f;
                w = 0f;
            }
            else
            {
                real_t sinAngle = Mathf.Sin(angle * 0.5f);
                real_t cosAngle = Mathf.Cos(angle * 0.5f);
                real_t s = sinAngle / d;

                x = axis.x * s;
                y = axis.y * s;
                z = axis.z * s;
                w = cosAngle;
            }
        }

        /// <summary>
        /// Composes these two quaternions by multiplying them together.
        /// This has the effect of rotating the second quaternion
        /// (the child) by the first quaternion (the parent).
        /// </summary>
        /// <param name="left">The parent quaternion.</param>
        /// <param name="right">The child quaternion.</param>
        /// <returns>The composed quaternion.</returns>
        public static Quat operator *(Quat left, Quat right)
        {
            return new Quat
            (
                (left.w * right.x) + (left.x * right.w) + (left.y * right.z) - (left.z * right.y),
                (left.w * right.y) + (left.y * right.w) + (left.z * right.x) - (left.x * right.z),
                (left.w * right.z) + (left.z * right.w) + (left.x * right.y) - (left.y * right.x),
                (left.w * right.w) - (left.x * right.x) - (left.y * right.y) - (left.z * right.z)
            );
        }

        /// <summary>
        /// Adds each component of the left <see cref="Quat"/>
        /// to the right <see cref="Quat"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression, such as approximating an intermediate
        /// rotation between two nearby rotations.
        /// </summary>
        /// <param name="left">The left quaternion to add.</param>
        /// <param name="right">The right quaternion to add.</param>
        /// <returns>The added quaternion.</returns>
        public static Quat operator +(Quat left, Quat right)
        {
            return new Quat(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
        }

        /// <summary>
        /// Subtracts each component of the left <see cref="Quat"/>
        /// by the right <see cref="Quat"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The left quaternion to subtract.</param>
        /// <param name="right">The right quaternion to subtract.</param>
        /// <returns>The subtracted quaternion.</returns>
        public static Quat operator -(Quat left, Quat right)
        {
            return new Quat(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w);
        }

        /// <summary>
        /// Returns the negative value of the <see cref="Quat"/>.
        /// This is the same as writing
        /// <c>new Quat(-q.x, -q.y, -q.z, -q.w)</c>. This operation
        /// results in a quaternion that represents the same rotation.
        /// </summary>
        /// <param name="quat">The quaternion to negate.</param>
        /// <returns>The negated quaternion.</returns>
        public static Quat operator -(Quat quat)
        {
            return new Quat(-quat.x, -quat.y, -quat.z, -quat.w);
        }

        [Obsolete("This operator does not have the correct behavior and will be replaced in the future. Do not use this.")]
        public static Quat operator *(Quat left, Vector3 right)
        {
            return new Quat
            (
                (left.w * right.x) + (left.y * right.z) - (left.z * right.y),
                (left.w * right.y) + (left.z * right.x) - (left.x * right.z),
                (left.w * right.z) + (left.x * right.y) - (left.y * right.x),
                -(left.x * right.x) - (left.y * right.y) - (left.z * right.z)
            );
        }

        [Obsolete("This operator does not have the correct behavior and will be replaced in the future. Do not use this.")]
        public static Quat operator *(Vector3 left, Quat right)
        {
            return new Quat
            (
                (right.w * left.x) + (right.y * left.z) - (right.z * left.y),
                (right.w * left.y) + (right.z * left.x) - (right.x * left.z),
                (right.w * left.z) + (right.x * left.y) - (right.y * left.x),
                -(right.x * left.x) - (right.y * left.y) - (right.z * left.z)
            );
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The quaternion to multiply.</param>
        /// <param name="right">The value to multiply by.</param>
        /// <returns>The multiplied quaternion.</returns>
        public static Quat operator *(Quat left, real_t right)
        {
            return new Quat(left.x * right, left.y * right, left.z * right, left.w * right);
        }

        /// <summary>
        /// Multiplies each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The value to multiply by.</param>
        /// <param name="right">The quaternion to multiply.</param>
        /// <returns>The multiplied quaternion.</returns>
        public static Quat operator *(real_t left, Quat right)
        {
            return new Quat(right.x * left, right.y * left, right.z * left, right.w * left);
        }

        /// <summary>
        /// Divides each component of the <see cref="Quat"/>
        /// by the given <see cref="real_t"/>. This operation is not
        /// meaningful on its own, but it can be used as a part of a
        /// larger expression.
        /// </summary>
        /// <param name="left">The quaternion to divide.</param>
        /// <param name="right">The value to divide by.</param>
        /// <returns>The divided quaternion.</returns>
        public static Quat operator /(Quat left, real_t right)
        {
            return left * (1.0f / right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the quaternions are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left quaternion.</param>
        /// <param name="right">The right quaternion.</param>
        /// <returns>Whether or not the quaternions are exactly equal.</returns>
        public static bool operator ==(Quat left, Quat right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the quaternions are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left quaternion.</param>
        /// <param name="right">The right quaternion.</param>
        /// <returns>Whether or not the quaternions are not equal.</returns>
        public static bool operator !=(Quat left, Quat right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// 如果此四元数和 <paramref name="obj"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="obj">要比较的另一个对象。</param>
        /// <returns>四元数和其他对象是否相等。</returns>
        public override bool Equals(object obj)
        {
            if (obj is Quat)
            {
                return Equals((Quat)obj);
            }

            return false;
        }

        /// <summary>
        /// 如果此四元数和 <paramref name="other"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="other">要比较的另一个四元数。</param>
        /// <returns>四元数是否相等。</returns>
        public bool Equals(Quat other)
        {
            return x == other.x && y == other.y && z == other.z && w == other.w;
        }

        /// <summary>
        /// 如果这个四元数和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Mathf.IsEqualApprox(real_t, real_t)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个四元数。</param>
        /// <returns>四元数是否近似相等。</returns>
        public bool IsEqualApprox(Quat other)
        {
            return Mathf.IsEqualApprox(x, other.x) && Mathf.IsEqualApprox(y, other.y) && Mathf.IsEqualApprox(z, other.z) && Mathf.IsEqualApprox(w, other.w);
        }

        /// <summary>
        /// 用作 <see cref="Quat"/> 的哈希函数。
        /// </summary>
        /// <returns>这个四元数的哈希码。</returns>
        public override int GetHashCode()
        {
            return y.GetHashCode() ^ x.GetHashCode() ^ z.GetHashCode() ^ w.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Quat"/> 转换为字符串。
        /// </summary>
        /// <returns>这个四元数的字符串表示。</returns>
        public override string ToString()
        {
            return $"({x}, {y}, {z}, {w})";
        }

        /// <summary>
        /// 将此 <see cref="Quat"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此四元数的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return $"({x.ToString(format)}, {y.ToString(format)}, {z.ToString(format)}, {w.ToString(format)})";
        }
    }
}
