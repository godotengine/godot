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
    /// 2×3 矩阵（2 行，3 列）用于 2D 线性变换。
    /// 可以表示平移、旋转或缩放等变换。
    /// 它由三个 <see cref="Vector2"/> 值组成：x、y 和原点。
    ///
    /// 有关更多信息，请阅读此文档文章：
    /// https://docs.godotengine.org/en/3.4/tutorials/math/matrices_and_transforms.html
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform2D : IEquatable<Transform2D>
    {
        /// <summary>
        /// 基矩阵的 X 向量（第 0 列）。 等效于数组索引 <c>[0]</c>。
        /// </summary>
        public Vector2 x;

        /// <summary>
        /// 基矩阵的 Y 向量（第 1 列）。 等效于数组索引<c>[1]</c>。
        /// </summary>
        public Vector2 y;

        /// <summary>
        /// 原点向量（第 2 列，第 3 列）。 等价于数组索引<c>[2]</c>。
        /// 原点向量代表平移。
        /// </summary>
        public Vector2 origin;

        /// <summary>
        /// 这个变换矩阵的旋转。
        /// </summary>
        /// <value>获取相当于用<see cref="x"/>的值调用<see cref="Mathf.Atan2(real_t, real_t)"/>。</value>
        public real_t Rotation
        {
            get
            {
                return Mathf.Atan2(x.y, x.x);
            }
            set
            {
                Vector2 scale = Scale;
                x.x = y.y = Mathf.Cos(value);
                x.y = y.x = Mathf.Sin(value);
                y.x *= -1;
                Scale = scale;
            }
        }

        /// <summary>
        /// 这个变换矩阵的尺度。
        /// </summary>
        /// <value>等价于每个列向量的长度，但是如果行列式为负，则Y为负。</value>
        public Vector2 Scale
        {
            get
            {
                real_t detSign = Mathf.Sign(BasisDeterminant());
                return new Vector2(x.Length(), detSign * y.Length());
            }
            set
            {
                value /= Scale; // Value becomes what's called "delta_scale" in core.
                x *= value.x;
                y *= value.y;
            }
        }

        /// <summary>
        /// 以 <see cref="Vector2"/> 的形式访问整个列。
        /// 第三列是 <see cref="origin"/> 向量。
        /// </summary>
        /// <param name="column">哪一列向量。</param>
        public Vector2 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return origin;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (column)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        origin = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// 以列优先顺序访问矩阵元素。
        /// 第三列是 <see cref="origin"/> 向量。
        /// </summary>
        /// <param name="column">哪一列，矩阵水平位置。</param>
        /// <param name="row">哪一行，矩阵垂直位置。</param>
        public real_t this[int column, int row]
        {
            get
            {
                return this[column][row];
            }
            set
            {
                Vector2 columnVector = this[column];
                columnVector[row] = value;
                this[column] = columnVector;
            }
        }

        /// <summary>
        /// 返回变换的逆，假设为
        /// 变换由旋转、缩放和平移组成。
        /// </summary>
        /// <seealso cref="Inverse"/>
        /// <returns>逆变换矩阵。</returns>
        public Transform2D AffineInverse()
        {
            real_t det = BasisDeterminant();

            if (det == 0)
                throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

            Transform2D inv = this;

            real_t temp = inv[0, 0];
            inv[0, 0] = inv[1, 1];
            inv[1, 1] = temp;

            real_t detInv = 1.0f / det;

            inv[0] *= new Vector2(detInv, -detInv);
            inv[1] *= new Vector2(-detInv, detInv);

            inv[2] = inv.BasisXform(-inv[2]);

            return inv;
        }

        /// <summary>
        /// 返回基矩阵的行列式。 如果基础是
        /// 统一缩放，其行列式是缩放的平方。
        ///
        /// 负行列式意味着 Y 比例为负。
        /// 零行列式意味着基不可逆，
        /// 并且通常被认为是无效的。
        /// </summary>
        /// <returns>基矩阵的行列式。</returns>
        private real_t BasisDeterminant()
        {
            return (x.x * y.y) - (x.y * y.x);
        }

        /// <summary>
        /// 返回由基矩阵变换（相乘）的向量。
        /// 此方法不考虑翻译（<see cref="origin"/> 向量）。
        /// </summary>
        /// <seealso cref="BasisXformInv(Vector2)"/>
        /// <param name="v">要转换的向量。</param>
        /// <returns>转换后的向量。</returns>
        public Vector2 BasisXform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v));
        }

        /// <summary>
        /// 返回由逆基矩阵变换（相乘）的向量。
        /// 此方法不考虑翻译（<see cref="origin"/> 向量）。
        ///
        /// 注意：这会导致乘以
        /// 仅当它表示旋转反射时的基矩阵。
        /// </summary>
        /// <seealso cref="BasisXform(Vector2)"/>
        /// <param name="v">要逆变换的向量。</param>
        /// <returns>逆变换后的向量。</returns>
        public Vector2 BasisXformInv(Vector2 v)
        {
            return new Vector2(x.Dot(v), y.Dot(v));
        }

        /// <summary>
        /// 通过 <paramref name="weight"/> 将此变换插入到另一个 <paramref name="transform"/>。
        /// </summary>
        /// <param name="transform">另一个变换。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值变换。</returns>
        public Transform2D InterpolateWith(Transform2D transform, real_t weight)
        {
            real_t r1 = Rotation;
            real_t r2 = transform.Rotation;

            Vector2 s1 = Scale;
            Vector2 s2 = transform.Scale;

            // Slerp rotation
            var v1 = new Vector2(Mathf.Cos(r1), Mathf.Sin(r1));
            var v2 = new Vector2(Mathf.Cos(r2), Mathf.Sin(r2));

            real_t dot = v1.Dot(v2);

            dot = Mathf.Clamp(dot, -1.0f, 1.0f);

            Vector2 v;

            if (dot > 0.9995f)
            {
                // Linearly interpolate to avoid numerical precision issues
                v = v1.LinearInterpolate(v2, weight).Normalized();
            }
            else
            {
                real_t angle = weight * Mathf.Acos(dot);
                Vector2 v3 = (v2 - (v1 * dot)).Normalized();
                v = (v1 * Mathf.Cos(angle)) + (v3 * Mathf.Sin(angle));
            }

            // Extract parameters
            Vector2 p1 = origin;
            Vector2 p2 = transform.origin;

            // Construct matrix
            var res = new Transform2D(Mathf.Atan2(v.y, v.x), p1.LinearInterpolate(p2, weight));
            Vector2 scale = s1.LinearInterpolate(s2, weight);
            res.x *= scale;
            res.y *= scale;

            return res;
        }

        /// <summary>
        /// 返回变换的逆，假设为
        /// 变换由旋转和平移组成
        /// （无缩放，使用 <see cref="AffineInverse"/> 进行缩放变换）。
        /// </summary>
        /// <returns>逆矩阵。</returns>
        public Transform2D Inverse()
        {
            Transform2D inv = this;

            // Swap
            real_t temp = inv.x.y;
            inv.x.y = inv.y.x;
            inv.y.x = temp;

            inv.origin = inv.BasisXform(-inv.origin);

            return inv;
        }

        /// <summary>
        /// 返回基础正交（90 度）的变换，
        /// 和归一化的轴向量（比例为 1 或 -1）。
        /// </summary>
        /// <returns>正交归一化变换。</returns>
        public Transform2D Orthonormalized()
        {
            Transform2D on = this;

            Vector2 onX = on.x;
            Vector2 onY = on.y;

            onX.Normalize();
            onY = onY - (onX * onX.Dot(onY));
            onY.Normalize();

            on.x = onX;
            on.y = onY;

            return on;
        }

        /// <summary>
        /// 使用矩阵乘法将变换旋转 <paramref name="phi"/>（以弧度为单位）。
        /// </summary>
        /// <param name="phi">要旋转的角度，以弧度为单位。</param>
        /// <returns>旋转后的变换矩阵。</returns>
        public Transform2D Rotated(real_t phi)
        {
            return this * new Transform2D(phi, new Vector2());
        }

        /// <summary>
        /// 使用矩阵乘法按给定的缩放因子缩放变换。
        /// </summary>
        /// <param name="scale">要引入的比例。</param>
        /// <returns>缩放的变换矩阵。</returns>
        public Transform2D Scaled(Vector2 scale)
        {
            Transform2D copy = this;
            copy.x *= scale;
            copy.y *= scale;
            copy.origin *= scale;
            return copy;
        }

        private void ScaleBasis(Vector2 scale)
        {
            x.x *= scale.x;
            x.y *= scale.y;
            y.x *= scale.x;
            y.y *= scale.y;
        }

        private real_t Tdotx(Vector2 with)
        {
            return (this[0, 0] * with[0]) + (this[1, 0] * with[1]);
        }

        private real_t Tdoty(Vector2 with)
        {
            return (this[0, 1] * with[0]) + (this[1, 1] * with[1]);
        }

        /// <summary>
        /// 通过给定的 <paramref name="offset"/> 翻译变换，
        /// 相对于变换的基向量。
        ///
        /// 与 <see cref="Rotated"/> 和 <see cref="Scaled"/> 不同，
        /// 这不使用矩阵乘法。
        /// </summary>
        /// <param name="offset">要翻译的偏移量。</param>
        /// <returns>翻译后的矩阵。</returns>
        public Transform2D Translated(Vector2 offset)
        {
            Transform2D copy = this;
            copy.origin += copy.BasisXform(offset);
            return copy;
        }

        /// <summary>
        /// 返回一个被这个变换矩阵变换（相乘）的向量。
        /// </summary>
        /// <seealso cref="XformInv(Vector2)"/>
        /// <param name="v">要转换的向量。</param>
        /// <returns>转换后的向量。</returns>
        [Obsolete("Xform is deprecated. Use the multiplication operator (Transform2D * Vector2) instead.")]
        public Vector2 Xform(Vector2 v)
        {
            return new Vector2(Tdotx(v), Tdoty(v)) + origin;
        }

        /// <summary>
        /// 返回由逆变换矩阵变换（相乘）的向量。
        /// </summary>
        /// <seealso cref="Xform(Vector2)"/>
        /// <param name="v">要逆变换的向量。</param>
        /// <returns>逆变换后的向量。</returns>
        [Obsolete("XformInv is deprecated. Use the multiplication operator (Vector2 * Transform2D) instead.")]
        public Vector2 XformInv(Vector2 v)
        {
            Vector2 vInv = v - origin;
            return new Vector2(x.Dot(vInv), y.Dot(vInv));
        }

        // Constants
        private static readonly Transform2D _identity = new Transform2D(1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipX = new Transform2D(-1, 0, 0, 1, 0, 0);
        private static readonly Transform2D _flipY = new Transform2D(1, 0, 0, -1, 0, 0);

        /// <summary>
        /// 恒等变换，没有应用平移、旋转或缩放。
        /// 这被用作 GDScript 中 <c>Transform2D()</c> 的替代品。
        /// 不要在 C# 中使用没有参数的 <c>new Transform2D()</c>，因为它会将所有值设置为零。
        /// </summary>
        /// <value>等价于<c>new Transform2D(Vector2.Right, Vector2.Down, Vector2.Zero)</c>.</value>
        public static Transform2D Identity { get { return _identity; } }
        /// <summary>
        /// 将沿 X 轴翻转某些东西的变换。
        /// </summary>
        /// <value>等价于<c>new Transform2D(Vector2.Left, Vector2.Down, Vector2.Zero)</c>.</value>
        public static Transform2D FlipX { get { return _flipX; } }
        /// <summary>
        /// 将沿 Y 轴翻转某些东西的变换。
        /// </summary>
        /// <value>等价于<c>new Transform2D(Vector2.Right, Vector2.Up, Vector2.Zero)</c>.</value>
        public static Transform2D FlipY { get { return _flipY; } }

        /// <summary>
        /// 从 3 个向量（矩阵列）构造一个变换矩阵。
        /// </summary>
        /// <param name="xAxis">X 向量，或列索引 0。</param>
        /// <param name="yAxis">Y 向量，或列索引 1。</param>
        /// <param name="originPos">原点向量，或列索引2。</param>
        public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 originPos)
        {
            x = xAxis;
            y = yAxis;
            origin = originPos;
        }

        /// <summary>
        /// 从给定的组件构造一个变换矩阵。
        /// 参数的命名使得 xy 等于调用 x.y
        /// </summary>
        /// <param name="xx">X列向量的X分量，通过<c>t.x.x</c>或<c>[0][0]</c></param>访问
        /// <param name="xy">X列向量的Y分量，通过<c>t.x.y</c>或<c>[0][1]</c></param>访问
        /// <param name="yx">Y列向量的X分量，通过<c>t.y.x</c>或<c>[1][0]</c></param>访问
        /// <param name="yy">Y列向量的Y分量，通过<c>t.y.y</c>或<c>[1][1]</c></param>访问
        /// <param name="ox">原点向量的X分量，通过<c>t.origin.x</c>或<c>[2][0]</c></param访问 >
        /// <param name="oy">原点向量的Y分量，通过<c>t.origin.y</c>或<c>[2][1]</c></param访问 >

        public Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy)
        {
            x = new Vector2(xx, xy);
            y = new Vector2(yx, yy);
            origin = new Vector2(ox, oy);
        }

        /// <summary>
        /// 从 <paramref name="rotation"/> 值和
        /// <paramref name="origin"/> vector.
        /// </summary>
        /// <param name="rotation">新变换的旋转，以弧度为单位。</param>
        /// <param name="origin">原点向量，或列索引2。</param>
        public Transform2D(real_t rotation, Vector2 origin)
        {
            x.x = y.y = Mathf.Cos(rotation);
            x.y = y.x = Mathf.Sin(rotation);
            y.x *= -1;
            this.origin = origin;
        }

        /// <summary>
        /// Composes these two transformation matrices by multiplying them
        /// together. This has the effect of transforming the second transform
        /// (the child) by the first transform (the parent).
        /// </summary>
        /// <param name="left">The parent transform.</param>
        /// <param name="right">The child transform.</param>
        /// <returns>The composed transform.</returns>
        public static Transform2D operator *(Transform2D left, Transform2D right)
        {
            left.origin = left * right.origin;

            real_t x0 = left.Tdotx(right.x);
            real_t x1 = left.Tdoty(right.x);
            real_t y0 = left.Tdotx(right.y);
            real_t y1 = left.Tdoty(right.y);

            left.x.x = x0;
            left.x.y = x1;
            left.y.x = y0;
            left.y.y = y1;

            return left;
        }

        /// <summary>
        /// 返回一个由变换矩阵变换（相乘）的 Vector2。
        /// </summary>
        /// <param name="transform">要应用的转换。</param>
        /// <param name="vector">要转换的 Vector2。</param>
        /// <returns>转换后的 Vector2.</returns>
        public static Vector2 operator *(Transform2D transform, Vector2 vector)
        {
            return new Vector2(transform.Tdotx(vector), transform.Tdoty(vector)) + transform.origin;
        }

        /// <summary>
        /// 返回由逆变换矩阵变换（相乘）的 Vector2。
        /// </summary>
        /// <param name="vector">要逆变换的Vector2。</param>
        /// <param name="transform">要应用的转换。</param>
        /// <returns>逆变换后的 Vector2.</returns>
        public static Vector2 operator *(Vector2 vector, Transform2D transform)
        {
            Vector2 vInv = vector - transform.origin;
            return new Vector2(transform.x.Dot(vInv), transform.y.Dot(vInv));
        }

        /// <summary>
        /// 返回一个由变换矩阵变换（相乘）的 Rect2。
        /// </summary>
        /// <param name="transform">要应用的转换。</param>
        /// <param name="rect">要转换的 Rect2。</param>
        /// <returns>转换后的 Rect2.</returns>
        public static Rect2 operator *(Transform2D transform, Rect2 rect)
        {
            Vector2 pos = transform * rect.Position;
            Vector2 toX = transform.x * rect.Size.x;
            Vector2 toY = transform.y * rect.Size.y;

            return new Rect2(pos, rect.Size).Expand(pos + toX).Expand(pos + toY).Expand(pos + toX + toY);
        }

        /// <summary>
        /// 返回一个由逆变换矩阵变换（相乘）的 Rect2。
        /// </summary>
        /// <param name="rect">一个要逆变换的Rect2。</param>
        /// <param name="transform">要应用的转换。</param>
        /// <returns>逆变换后的 Rect2.</returns>
        public static Rect2 operator *(Rect2 rect, Transform2D transform)
        {
            Vector2 pos = rect.Position * transform;
            Vector2 to1 = new Vector2(rect.Position.x, rect.Position.y + rect.Size.y) * transform;
            Vector2 to2 = new Vector2(rect.Position.x + rect.Size.x, rect.Position.y + rect.Size.y) * transform;
            Vector2 to3 = new Vector2(rect.Position.x + rect.Size.x, rect.Position.y) * transform;

            return new Rect2(pos, rect.Size).Expand(to1).Expand(to2).Expand(to3);
        }

        /// <summary>
        /// 返回由变换矩阵变换（相乘）的给定 Vector2[] 的副本。
        /// </summary>
        /// <param name="transform">要应用的转换。</param>
        /// <param name="array">要转换的 Vector2[]。</param>
        /// <returns>Vector2[] 的转换后的副本。</returns>
        public static Vector2[] operator *(Transform2D transform, Vector2[] array)
        {
            Vector2[] newArray = new Vector2[array.Length];

            for (int i = 0; i < array.Length; i++)
            {
                newArray[i] = transform * array[i];
            }

            return newArray;
        }

        /// <summary>
        /// 返回由逆变换矩阵变换（相乘）的给定 Vector2[] 的副本。
        /// </summary>
        /// <param name="array">一个 Vector2[] 进行逆变换。</param>
        /// <param name="transform">要应用的转换。</param>
        /// <returns>Vector2[] 的逆变换副本。</returns>
        public static Vector2[] operator *(Vector2[] array, Transform2D transform)
        {
            Vector2[] newArray = new Vector2[array.Length];

            for (int i = 0; i < array.Length; i++)
            {
                newArray[i] = array[i] * transform;
            }

            return newArray;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the transforms are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left transform.</param>
        /// <param name="right">The right transform.</param>
        /// <returns>Whether or not the transforms are exactly equal.</returns>
        public static bool operator ==(Transform2D left, Transform2D right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the transforms are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left transform.</param>
        /// <param name="right">The right transform.</param>
        /// <returns>Whether or not the transforms are not equal.</returns>
        public static bool operator !=(Transform2D left, Transform2D right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// 如果此转换和 <paramref name="obj"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="obj">要比较的另一个对象。</param>
        /// <returns>变换和其他对象是否相等。</returns>
        public override bool Equals(object obj)
        {
            return obj is Transform2D transform2D && Equals(transform2D);
        }

        /// <summary>
        /// 如果此转换和 <paramref name="other"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="other">要比较的另一个变换。</param>
        /// <returns>矩阵是否相等。</returns>
        public bool Equals(Transform2D other)
        {
            return x.Equals(other.x) && y.Equals(other.y) && origin.Equals(other.origin);
        }

        /// <summary>
        /// 如果此变换和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Vector2.IsEqualApprox(Vector2)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个变换。</param>
        /// <returns>矩阵是否近似相等。</returns>
        public bool IsEqualApprox(Transform2D other)
        {
            return x.IsEqualApprox(other.x) && y.IsEqualApprox(other.y) && origin.IsEqualApprox(other.origin);
        }

        /// <summary>
        /// 用作 <see cref="Transform2D"/> 的哈希函数。
        /// </summary>
        /// <returns>此转换的哈希码。</returns>
        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ origin.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Transform2D"/> 转换为字符串。
        /// </summary>
        /// <returns>此转换的字符串表示形式。</returns>
        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(),
                y.ToString(),
                origin.ToString()
            });
        }

        /// <summary>
        /// 将此 <see cref="Transform2D"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此转换的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                x.ToString(format),
                y.ToString(format),
                origin.ToString(format)
            });
        }
    }
}
