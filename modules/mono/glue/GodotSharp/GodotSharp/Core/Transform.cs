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
    /// 3×4 矩阵（3 行，4 列）用于 3D 线性变换。
    /// 可以表示平移、旋转或缩放等变换。
    /// 它由一个 <see cref="Basis"/>（前 3 列）和一个
    /// <see cref="Vector3"/> 对于原点（最后一列）。
    ///
    /// 有关更多信息，请阅读此文档文章：
    /// https://docs.godotengine.org/en/3.4/tutorials/math/matrices_and_transforms.html
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Transform : IEquatable<Transform>
    {
        /// <summary>
        /// 此转换的 <see cref="Basis"/>。 包含 X、Y 和 Z 基
        /// 向量（第 0 到 2 列）并负责旋转和缩放。
        /// </summary>
        public Basis basis;

        /// <summary>
        /// 原点向量（第 3 列，第 4 列）。 等价于数组索引<c>[3]</c>。
        /// </summary>
        public Vector3 origin;

        /// <summary>
        /// 以 <see cref="Vector3"/> 的形式访问整个列。
        /// 第四列是 <see cref="origin"/> 向量。
        /// </summary>
        /// <param name="column">Which column vector.</param>
        public Vector3 this[int column]
        {
            get
            {
                switch (column)
                {
                    case 0:
                        return basis.Column0;
                    case 1:
                        return basis.Column1;
                    case 2:
                        return basis.Column2;
                    case 3:
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
                        basis.Column0 = value;
                        return;
                    case 1:
                        basis.Column1 = value;
                        return;
                    case 2:
                        basis.Column2 = value;
                        return;
                    case 3:
                        origin = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        /// <summary>
        /// 以列优先顺序访问矩阵元素。
        /// 第四列是 <see cref="origin"/> 向量。
        /// </summary>
        /// <param name="column">哪一列，矩阵水平位置。</param>
        /// <param name="row">哪一行，矩阵垂直位置。</param>
        public real_t this[int column, int row]
        {
            get
            {
                if (column == 3)
                {
                    return origin[row];
                }
                return basis[column, row];
            }
            set
            {
                if (column == 3)
                {
                    origin[row] = value;
                    return;
                }
                basis[column, row] = value;
            }
        }

        /// <summary>
        /// 返回变换的逆，假设为
        /// 变换由旋转、缩放和平移组成。
        /// </summary>
        /// <seealso cref="Inverse"/>
        /// <returns>逆变换矩阵。</returns>
        public Transform AffineInverse()
        {
            Basis basisInv = basis.Inverse();
            return new Transform(basisInv, basisInv.Xform(-origin));
        }

        /// <summary>
        /// 通过 <paramref name="weight"/> 将此变换插入到另一个 <paramref name="transform"/>。
        /// </summary>
        /// <param name="transform">另一个变换。</param>
        /// <param name="weight">0.0到1.0范围内的一个值，代表插值量。</param>
        /// <returns>插值变换。</returns>
        public Transform InterpolateWith(Transform transform, real_t weight)
        {
            /* not sure if very "efficient" but good enough? */

            Vector3 sourceScale = basis.Scale;
            Quat sourceRotation = basis.RotationQuat();
            Vector3 sourceLocation = origin;

            Vector3 destinationScale = transform.basis.Scale;
            Quat destinationRotation = transform.basis.RotationQuat();
            Vector3 destinationLocation = transform.origin;

            var interpolated = new Transform();
            interpolated.basis.SetQuatScale(sourceRotation.Slerp(destinationRotation, weight).Normalized(), sourceScale.LinearInterpolate(destinationScale, weight));
            interpolated.origin = sourceLocation.LinearInterpolate(destinationLocation, weight);

            return interpolated;
        }

        /// <summary>
        /// 返回变换的逆，假设为
        /// 变换由旋转和平移组成
        /// （无缩放，使用 <see cref="AffineInverse"/> 进行缩放变换）。
        /// </summary>
        /// <returns>逆矩阵。</returns>
        public Transform Inverse()
        {
            Basis basisTr = basis.Transposed();
            return new Transform(basisTr, basisTr.Xform(-origin));
        }

        /// <summary>
        /// 返回变换的副本，使其旋转
        /// -Z 轴（向前）指向 <paramref name="target"/> 位置。
        ///
        /// 变换将首先围绕给定的 <paramref name="up"/> 向量旋转，
        /// 然后通过进一步旋转完全对齐到 <paramref name="target"/>
        /// 垂直于 <paramref name="target"/> 和 <paramref name="up"/> 向量的轴。
        ///
        /// 操作发生在全局空间中。
        /// </summary>
        /// <param name="target">要查看的对象。</param>
        /// <param name="up">相对向上方向</param>
        /// <returns>生成的转换。</returns>
        public Transform LookingAt(Vector3 target, Vector3 up)
        {
            Transform t = this;
            t.SetLookAt(origin, target, up);
            return t;
        }

        /// <summary>
        /// 返回基础正交（90 度）的变换，
        /// 和归一化的轴向量（比例为 1 或 -1）。
        /// </summary>
        /// <returns>正交归一化变换。</returns>
        public Transform Orthonormalized()
        {
            return new Transform(basis.Orthonormalized(), origin);
        }

        /// <summary>
        /// 围绕给定的 <paramref name="axis"/> 旋转变换 <paramref name="phi"/>（以弧度为单位），
        /// 使用矩阵乘法。 轴必须是归一化向量。
        /// </summary>
        /// <param name="axis">要旋转的轴。 必须标准化。</param>
        /// <param name="phi">要旋转的角度，以弧度为单位。</param>
        /// <returns>旋转后的变换矩阵。</returns>

        public Transform Rotated(Vector3 axis, real_t phi)
        {
            return new Transform(new Basis(axis, phi), new Vector3()) * this;
        }

        /// <summary>
        /// 通过给定的 3D 缩放因子缩放变换，使用矩阵乘法。
        /// </summary>
        /// <param name="scale">要引入的比例。</param>
        /// <returns>缩放的变换矩阵。</returns>
        public Transform Scaled(Vector3 scale)
        {
            return new Transform(basis.Scaled(scale), origin * scale);
        }

        private void SetLookAt(Vector3 eye, Vector3 target, Vector3 up)
        {
            // Make rotation matrix
            // Z vector
            Vector3 column2 = eye - target;

            column2.Normalize();

            Vector3 column1 = up;

            Vector3 column0 = column1.Cross(column2);

            // Recompute Y = Z cross X
            column1 = column2.Cross(column0);

            column0.Normalize();
            column1.Normalize();

            basis = new Basis(column0, column1, column2);

            origin = eye;
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
        public Transform Translated(Vector3 offset)
        {
            return new Transform(basis, new Vector3
            (
                origin[0] += basis.Row0.Dot(offset),
                origin[1] += basis.Row1.Dot(offset),
                origin[2] += basis.Row2.Dot(offset)
            ));
        }

        /// <summary>
        /// 返回一个被这个变换矩阵变换（相乘）的向量。
        /// </summary>
        /// <seealso cref="XformInv(Vector3)"/>
        /// <param name="v">要转换的向量。</param>
        /// <returns>转换后的向量。</returns>
        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                basis.Row0.Dot(v) + origin.x,
                basis.Row1.Dot(v) + origin.y,
                basis.Row2.Dot(v) + origin.z
            );
        }

        /// <summary>
        /// 返回由转置变换矩阵变换（相乘）的向量。
        ///
        /// 注意：这会导致乘以
        /// 仅当它表示旋转反射时的变换矩阵。
        /// </summary>
        /// <seealso cref="Xform(Vector3)"/>
        /// <param name="v">要逆变换的向量。</param>
        /// <returns>逆变换后的向量。</returns>
        public Vector3 XformInv(Vector3 v)
        {
            Vector3 vInv = v - origin;

            return new Vector3
            (
                (basis.Row0[0] * vInv.x) + (basis.Row1[0] * vInv.y) + (basis.Row2[0] * vInv.z),
                (basis.Row0[1] * vInv.x) + (basis.Row1[1] * vInv.y) + (basis.Row2[1] * vInv.z),
                (basis.Row0[2] * vInv.x) + (basis.Row1[2] * vInv.y) + (basis.Row2[2] * vInv.z)
            );
        }

        // Constants
        private static readonly Transform _identity = new Transform(Basis.Identity, Vector3.Zero);
        private static readonly Transform _flipX = new Transform(new Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1), Vector3.Zero);
        private static readonly Transform _flipY = new Transform(new Basis(1, 0, 0, 0, -1, 0, 0, 0, 1), Vector3.Zero);
        private static readonly Transform _flipZ = new Transform(new Basis(1, 0, 0, 0, 1, 0, 0, 0, -1), Vector3.Zero);

        /// <summary>
        /// 恒等变换，没有应用平移、旋转或缩放。
        /// 这被用作 GDScript 中 <c>Transform()</c> 的替代品。
        /// 不要在 C# 中使用没有参数的 <c>new Transform()</c>，因为它会将所有值设置为零。
        /// </summary>
        /// <value>相当于<c>new Transform(Vector3.Right, Vector3.Up, Vector3.Back, Vector3.Zero)</c>.</value>
        public static Transform Identity { get { return _identity; } }
        /// <summary>
        /// 将沿 X 轴翻转某些东西的变换。
        /// </summary>
        /// <value>等价于<c>new Transform(Vector3.Left, Vector3.Up, Vector3.Back, Vector3.Zero)</c>.</value>
        public static Transform FlipX { get { return _flipX; } }
        /// <summary>
        /// 将沿 Y 轴翻转某些东西的变换。
        /// </summary>
        /// <value>等价于<c>new Transform(Vector3.Right, Vector3.Down, Vector3.Back, Vector3.Zero)</c>.</value>
        public static Transform FlipY { get { return _flipY; } }
        /// <summary>
        /// 将沿 Z 轴翻转某些东西的变换。
        /// </summary>
        /// <value>相当于<c>new Transform(Vector3.Right, Vector3.Up, Vector3.Forward, Vector3.Zero)</c>.</value>
        public static Transform FlipZ { get { return _flipZ; } }

        /// <summary>
        /// 从 4 个向量（矩阵列）构造一个变换矩阵。
        /// </summary>
        /// <param name="column0">X 向量，或列索引 0。</param>
        /// <param name="column1">Y 向量，或列索引 1。</param>
        /// <param name="column2">Z 向量，或列索引 2。</param>
        /// <param name="origin">原点向量，或列索引3。</param>
        public Transform(Vector3 column0, Vector3 column1, Vector3 column2, Vector3 origin)
        {
            basis = new Basis(column0, column1, column2);
            this.origin = origin;
        }

        /// <summary>
        /// 从给定的 <paramref name="quaternion"/> 构造一个变换矩阵
        /// and <paramref name="origin"/> vector.
        /// </summary>
        /// <param name="quaternion">创建基础的<see cref="Quat"/>。</param>
        /// <param name="origin">原点向量，或列索引3。</param>
        public Transform(Quat quaternion, Vector3 origin)
        {
            basis = new Basis(quaternion);
            this.origin = origin;
        }

        /// <summary>
        /// 从给定的 <paramref name="basis"/> 和
        /// <paramref name="origin"/> 向量。
        /// </summary>
        /// <param name="basis">创建基础的<see cref="Basis"/>。</param>
        /// <param name="origin">原点向量，或列索引3。</param>
        public Transform(Basis basis, Vector3 origin)
        {
            this.basis = basis;
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
        public static Transform operator *(Transform left, Transform right)
        {
            left.origin = left.Xform(right.origin);
            left.basis *= right.basis;
            return left;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the transforms are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left transform.</param>
        /// <param name="right">The right transform.</param>
        /// <returns>Whether or not the transforms are exactly equal.</returns>
        public static bool operator ==(Transform left, Transform right)
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
        public static bool operator !=(Transform left, Transform right)
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
            if (obj is Transform)
            {
                return Equals((Transform)obj);
            }

            return false;
        }

        /// <summary>
        /// 如果此转换和 <paramref name="other"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="other">要比较的另一个变换。</param>
        /// <returns>矩阵是否相等。</returns>
        public bool Equals(Transform other)
        {
            return basis.Equals(other.basis) && origin.Equals(other.origin);
        }

        /// <summary>
        /// 如果此变换和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Vector3.IsEqualApprox(Vector3)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个变换。</param>
        /// <returns>矩阵是否近似相等。</returns>
        public bool IsEqualApprox(Transform other)
        {
            return basis.IsEqualApprox(other.basis) && origin.IsEqualApprox(other.origin);
        }

        /// <summary>
        /// 用作 <see cref="Transform"/> 的散列函数。
        /// </summary>
        /// <returns>此转换的哈希码。</returns>
        public override int GetHashCode()
        {
            return basis.GetHashCode() ^ origin.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Transform"/> 转换为字符串。
        /// </summary>
        /// <returns>此转换的字符串表示形式。</returns>
        public override string ToString()
        {
            return String.Format("{0} - {1}", new object[]
            {
                basis.ToString(),
                origin.ToString()
            });
        }

        /// <summary>
        /// 将此 <see cref="Transform"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此转换的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return String.Format("{0} - {1}", new object[]
            {
                basis.ToString(format),
                origin.ToString(format)
            });
        }
    }
}
