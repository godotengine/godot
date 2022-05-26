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
    /// 2D 轴对齐边界框。 Rect2 由位置、大小和
    /// 几个实用函数。 它通常用于快速重叠测试。
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2 : IEquatable<Rect2>
    {
        private Vector2 _position;
        private Vector2 _size;

        /// <summary>
        /// 起点角。 通常具有低于 <see cref="End"/> 的值。
        /// </summary>
        /// <value>直接使用私有字段。</value>
        public Vector2 Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// 从 <see cref="Position"/> 到 <see cref="End"/> 的大小。 通常所有成分都是阳性的。
        /// 如果size是负数，可以使用<see cref="Abs"/>来修复。
        /// </summary>
        /// <value>直接使用私有字段。</value>
        public Vector2 Size
        {
            get { return _size; }
            set { _size = value; }
        }

        /// <summary>
        /// 结束角。 这计算为 <see cref="Position"/> 加上 <see cref="Size"/>。
        /// 设置这个值会改变大小。
        /// </summary>
        /// <value>
        /// 获取等价于 <paramref name="value"/> = <see cref="Position"/> + <see cref="Size"/>,
        /// 设置等价于 <see cref="Size"/> = <paramref name="value"/> - <see cref="Position"/>
        /// </value>
        public Vector2 End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// 这个 <see cref="Rect2"/> 的区域。
        /// </summary>
        /// <value>等价于<see cref="GetArea()"/>。</value>
        public real_t Area
        {
            get { return GetArea(); }
        }

        /// <summary>
        /// 返回一个 <see cref="Rect2"/> 具有相同的位置和大小，修改后
        /// 左上角为原点，宽高为正。
        /// </summary>
        /// <returns>修改后的<see cref="Rect2"/>.</returns>
        public Rect2 Abs()
        {
            Vector2 end = End;
            Vector2 topLeft = new Vector2(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2(topLeft, _size.Abs());
        }

        /// <summary>
        /// 返回此 <see cref="Rect2"/> 和 <paramref name="b"/> 的交集。
        /// 如果矩形不相交，则返回一个空的 <see cref="Rect2"/>。
        /// </summary>
        /// <param name="b">另一个<see cref="Rect2"/>.</param>
        /// <returns>被剪裁的<see cref="Rect2"/>.</returns>
        public Rect2 Clip(Rect2 b)
        {
            Rect2 newRect = b;

            if (!Intersects(newRect))
            {
                return new Rect2();
            }

            newRect._position.x = Mathf.Max(b._position.x, _position.x);
            newRect._position.y = Mathf.Max(b._position.y, _position.y);

            Vector2 bEnd = b._position + b._size;
            Vector2 end = _position + _size;

            newRect._size.x = Mathf.Min(bEnd.x, end.x) - newRect._position.x;
            newRect._size.y = Mathf.Min(bEnd.y, end.y) - newRect._position.y;

            return newRect;
        }

        /// <summary>
        /// 如果此 <see cref="Rect2"/> 完全包含另一个，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="b">另一个可能包含的<see cref="Rect2"/>。</param>
        /// <returns>
        /// 一个 <see langword="bool"/> 判断这个 <see cref="Rect2"/> 是否包含 <paramref name="b"/>。
        /// </returns>
        public bool Encloses(Rect2 b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        /// <summary>
        /// 返回此 <see cref="Rect2"/> 扩展为包含给定点。
        /// </summary>
        /// <param name="to">要包含的点。</param>
        /// <returns>扩展的<see cref="Rect2"/>.</returns>
        public Rect2 Expand(Vector2 to)
        {
            Rect2 expanded = this;

            Vector2 begin = expanded._position;
            Vector2 end = expanded._position + expanded._size;

            if (to.x < begin.x)
            {
                begin.x = to.x;
            }
            if (to.y < begin.y)
            {
                begin.y = to.y;
            }

            if (to.x > end.x)
            {
                end.x = to.x;
            }
            if (to.y > end.y)
            {
                end.y = to.y;
            }

            expanded._position = begin;
            expanded._size = end - begin;

            return expanded;
        }

        /// <summary>
        /// 返回 <see cref="Rect2"/> 的面积。
        /// </summary>
        /// <returns>区域。</returns>
        public real_t GetArea()
        {
            return _size.x * _size.y;
        }

        /// <summary>
        /// Returns the center of the <see cref="Rect2"/>, which is equal
        /// to <see cref="Position"/> + (<see cref="Size"/> / 2).
        /// </summary>
        /// <returns>The center.</returns>
        public Vector2 GetCenter()
        {
            return _position + (_size * 0.5f);
        }

        /// <summary>
        /// 返回一个 <see cref="Rect2"/> 的副本，其增长了给定数量的单位
        ///所有方面。
        /// </summary>
        /// <seealso cref="GrowIndividual(real_t, real_t, real_t, real_t)"/>
        /// <seealso cref="GrowMargin(Margin, real_t)"/>
        /// <param name="by">增长的数量。</param>
        /// <returns>长大的<see cref="Rect2"/>.</returns>
        public Rect2 Grow(real_t by)
        {
            Rect2 g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        /// <summary>
        /// 返回一个 <see cref="Rect2"/> 的副本，其增长了给定数量的单位
        /// 每个方向单独。
        /// </summary>
        /// <seealso cref="Grow(real_t)"/>
        /// <seealso cref="GrowMargin(Margin, real_t)"/>
        /// <param name="left">左边增长的数量。</param>
        /// <param name="top">顶部增长的数量。</param>
        /// <param name="right">右边增长的数量。</param>
        /// <param name="bottom">底部增长的数量。</param>
        /// <returns>长大的<see cref="Rect2"/>.</returns>
        public Rect2 GrowIndividual(real_t left, real_t top, real_t right, real_t bottom)
        {
            Rect2 g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        /// <summary>
        /// 返回一个 <see cref="Rect2"/> 的副本，其增长了给定数量的单位
        /// <see cref="Margin"/> 方向。
        /// </summary>
        /// <seealso cref="Grow(real_t)"/>
        /// <seealso cref="GrowIndividual(real_t, real_t, real_t, real_t)"/>
        /// <param name="margin">增长方向</param>
        /// <param name="by">增长的数量。</param>
        /// <returns>长大的<see cref="Rect2"/>.</returns>
        public Rect2 GrowMargin(Margin margin, real_t by)
        {
            Rect2 g = this;

            g = g.GrowIndividual(Margin.Left == margin ? by : 0,
                    Margin.Top == margin ? by : 0,
                    Margin.Right == margin ? by : 0,
                    Margin.Bottom == margin ? by : 0);

            return g;
        }

        /// <summary>
        /// 如果 <see cref="Rect2"/> 是平的或空的，则返回 <see langword="true"/>，
        /// 或 <see langword="false"/> 否则。
        /// </summary>
        /// <returns>
        /// <see langword="bool"/> 判断 <see cref="Rect2"/> 是否有面积。
        /// </returns>
        public bool HasNoArea()
        {
            return _size.x <= 0 || _size.y <= 0;
        }

        /// <summary>
        /// 如果 <see cref="Rect2"/> 包含一个点，则返回 <see langword="true"/>，
        /// 或 <see langword="false"/> 否则。
        /// </summary>
        /// <param name="point">要检查的点。</param>
        /// <返回>
        /// <see langword="bool"/> 判断 <see cref="Rect2"/> 是否包含 <paramref name="point"/>。
        /// </returns>
        public bool HasPoint(Vector2 point)
        {
            if (point.x < _position.x)
                return false;
            if (point.y < _position.y)
                return false;

            if (point.x >= _position.x + _size.x)
                return false;
            if (point.y >= _position.y + _size.y)
                return false;

            return true;
        }

        /// <summary>
        /// 如果 <see cref="Rect2"/> 与 <paramref name="b"/> 重叠，则返回 <see langword="true"/>
        ///（即它们至少有一个共同点）。
        ///
        /// 如果 <paramref name="includeBorders"/> 是 <see langword="true"/>,
        /// 如果它们的边框接触，它们也将被认为是重叠的，
        /// 即使没有交集。
        /// </summary>
        /// <param name="b">另一个 <see cref="Rect2"/> 来检查交叉点。</param>
        /// <param name="includeBorders">是否考虑边框</param>
        /// <returns>一个 <see langword="bool"/> 它们是否相交。</returns>
        public bool Intersects(Rect2 b, bool includeBorders = false)
        {
            if (includeBorders)
            {
                if (_position.x > b._position.x + b._size.x)
                {
                    return false;
                }
                if (_position.x + _size.x < b._position.x)
                {
                    return false;
                }
                if (_position.y > b._position.y + b._size.y)
                {
                    return false;
                }
                if (_position.y + _size.y < b._position.y)
                {
                    return false;
                }
            }
            else
            {
                if (_position.x >= b._position.x + b._size.x)
                {
                    return false;
                }
                if (_position.x + _size.x <= b._position.x)
                {
                    return false;
                }
                if (_position.y >= b._position.y + b._size.y)
                {
                    return false;
                }
                if (_position.y + _size.y <= b._position.y)
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// 返回一个更大的 <see cref="Rect2"/> 包含这个 <see cref="Rect2"/> 和 <paramref name="b"/>。
        /// </summary>
        /// <param name="b">另一个<see cref="Rect2"/>.</param>
        /// <returns>合并后的<see cref="Rect2"/>.</returns>
        public Rect2 Merge(Rect2 b)
        {
            Rect2 newRect;

            newRect._position.x = Mathf.Min(b._position.x, _position.x);
            newRect._position.y = Mathf.Min(b._position.y, _position.y);

            newRect._size.x = Mathf.Max(b._position.x + b._size.x, _position.x + _size.x);
            newRect._size.y = Mathf.Max(b._position.y + b._size.y, _position.y + _size.y);

            newRect._size -= newRect._position; // Make relative again

            return newRect;
        }

        /// <summary>
        /// 根据位置和大小构造一个 <see cref="Rect2"/>。
        /// </summary>
        /// <param name="position">位置。</param>
        /// <param name="size">大小。</param>
        public Rect2(Vector2 position, Vector2 size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// 根据位置、宽度和高度构造一个 <see cref="Rect2"/>。
        /// </summary>
        /// <param name="position">位置。</param>
        /// <param name="width">宽度。</param>
        /// <param name="height">高度。</param>
        public Rect2(Vector2 position, real_t width, real_t height)
        {
            _position = position;
            _size = new Vector2(width, height);
        }

        /// <summary>
        /// 从 x、y 和大小构造一个 <see cref="Rect2"/>。
        /// </summary>
        /// <param name="x">位置的X坐标</param>
        /// <param name="y">位置的Y坐标</param>
        /// <param name="size">大小。</param>
        public Rect2(real_t x, real_t y, Vector2 size)
        {
            _position = new Vector2(x, y);
            _size = size;
        }

        /// <summary>
        /// 从 x、y、宽度和高度构造一个 <see cref="Rect2"/>。
        /// </summary>
        /// <param name="x">位置的X坐标</param>
        /// <param name="y">位置的Y坐标</param>
        /// <param name="width">宽度。</param>
        /// <param name="height">高度。</param>
        public Rect2(real_t x, real_t y, real_t width, real_t height)
        {
            _position = new Vector2(x, y);
            _size = new Vector2(width, height);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2"/>s are exactly equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are exactly equal.</returns>
        public static bool operator ==(Rect2 left, Rect2 right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2"/>s are not equal.
        /// Note: Due to floating-point precision errors, consider using
        /// <see cref="IsEqualApprox"/> instead, which is more reliable.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are not equal.</returns>
        public static bool operator !=(Rect2 left, Rect2 right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// 如果此 rect 和 <paramref name="obj"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="obj">要比较的另一个对象。</param>
        /// <returns>rect和其他对象是否相等。</returns>
        public override bool Equals(object obj)
        {
            if (obj is Rect2)
            {
                return Equals((Rect2)obj);
            }

            return false;
        }

        /// <summary>
        /// 如果此 rect 和 <paramref name="other"/> 相等，则返回 <see langword="true"/>。
        /// </summary>
        /// <param name="other">要比较的另一个矩形。</param>
        /// <returns>矩形是否相等。</returns>
        public bool Equals(Rect2 other)
        {
            return _position.Equals(other._position) && _size.Equals(other._size);
        }

        /// <summary>
        /// 如果这个 rect 和 <paramref name="other"/> 近似相等，则返回 <see langword="true"/>，
        /// 通过在每个组件上运行 <see cref="Vector2.IsEqualApprox(Vector2)"/>。
        /// </summary>
        /// <param name="other">要比较的另一个矩形。</param>
        /// <returns>矩形是否近似相等。</returns>
        public bool IsEqualApprox(Rect2 other)
        {
            return _position.IsEqualApprox(other._position) && _size.IsEqualApprox(other.Size);
        }

        /// <summary>
        /// 用作 <see cref="Rect2"/> 的哈希函数。
        /// </summary>
        /// <returns>这个矩形的哈希码。</returns>
        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        /// <summary>
        /// 将此 <see cref="Rect2"/> 转换为字符串。
        /// </summary>
        /// <returns>此矩形的字符串表示形式。</returns>
        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                _position.ToString(),
                _size.ToString()
            });
        }

        /// <summary>
        /// 将此 <see cref="Rect2"/> 转换为具有给定 <paramref name="format"/> 的字符串。
        /// </summary>
        /// <returns>此矩形的字符串表示形式。</returns>
        public string ToString(string format)
        {
            return String.Format("({0}, {1})", new object[]
            {
                _position.ToString(format),
                _size.ToString(format)
            });
        }
    }
}
