using System;
using System.Runtime.InteropServices;

namespace Godot
{
    /// <summary>
    /// 2D axis-aligned bounding box using integers. Rect2i consists of a position, a size, and
    /// several utility functions. It is typically used for fast overlap tests.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2i : IEquatable<Rect2i>
    {
        private Vector2i _position;
        private Vector2i _size;

        /// <summary>
        /// Beginning corner. Typically has values lower than <see cref="End"/>.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2i Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from <see cref="Position"/> to <see cref="End"/>. Typically all components are positive.
        /// If the size is negative, you can use <see cref="Abs"/> to fix it.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2i Size
        {
            get { return _size; }
            set { _size = value; }
        }

        /// <summary>
        /// Ending corner. This is calculated as <see cref="Position"/> plus <see cref="Size"/>.
        /// Setting this value will change the size.
        /// </summary>
        /// <value>
        /// Getting is equivalent to <paramref name="value"/> = <see cref="Position"/> + <see cref="Size"/>,
        /// setting is equivalent to <see cref="Size"/> = <paramref name="value"/> - <see cref="Position"/>
        /// </value>
        public Vector2i End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// The area of this <see cref="Rect2i"/>.
        /// </summary>
        /// <value>Equivalent to <see cref="GetArea()"/>.</value>
        public int Area
        {
            get { return GetArea(); }
        }

        /// <summary>
        /// Returns a <see cref="Rect2i"/> with equivalent position and size, modified so that
        /// the top-left corner is the origin and width and height are positive.
        /// </summary>
        /// <returns>The modified <see cref="Rect2i"/>.</returns>
        public Rect2i Abs()
        {
            Vector2i end = End;
            Vector2i topLeft = new Vector2i(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2i(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns the intersection of this <see cref="Rect2i"/> and <paramref name="b"/>.
        /// If the rectangles do not intersect, an empty <see cref="Rect2i"/> is returned.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2i"/>.</param>
        /// <returns>
        /// The intersection of this <see cref="Rect2i"/> and <paramref name="b"/>,
        /// or an empty <see cref="Rect2i"/> if they do not intersect.
        /// </returns>
        public Rect2i Intersection(Rect2i b)
        {
            Rect2i newRect = b;

            if (!Intersects(newRect))
            {
                return new Rect2i();
            }

            newRect._position.x = Mathf.Max(b._position.x, _position.x);
            newRect._position.y = Mathf.Max(b._position.y, _position.y);

            Vector2i bEnd = b._position + b._size;
            Vector2i end = _position + _size;

            newRect._size.x = Mathf.Min(bEnd.x, end.x) - newRect._position.x;
            newRect._size.y = Mathf.Min(bEnd.y, end.y) - newRect._position.y;

            return newRect;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this <see cref="Rect2i"/> completely encloses another one.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2i"/> that may be enclosed.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not this <see cref="Rect2i"/> encloses <paramref name="b"/>.
        /// </returns>
        public bool Encloses(Rect2i b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        /// <summary>
        /// Returns this <see cref="Rect2i"/> expanded to include a given point.
        /// </summary>
        /// <param name="to">The point to include.</param>
        /// <returns>The expanded <see cref="Rect2i"/>.</returns>
        public Rect2i Expand(Vector2i to)
        {
            Rect2i expanded = this;

            Vector2i begin = expanded._position;
            Vector2i end = expanded._position + expanded._size;

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
        /// Returns the area of the <see cref="Rect2i"/>.
        /// </summary>
        /// <returns>The area.</returns>
        public int GetArea()
        {
            return _size.x * _size.y;
        }

        /// <summary>
        /// Returns the center of the <see cref="Rect2i"/>, which is equal
        /// to <see cref="Position"/> + (<see cref="Size"/> / 2).
        /// If <see cref="Size"/> is an odd number, the returned center
        /// value will be rounded towards <see cref="Position"/>.
        /// </summary>
        /// <returns>The center.</returns>
        public Vector2i GetCenter()
        {
            return _position + (_size / 2);
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2i"/> grown by the specified amount
        /// on all sides.
        /// </summary>
        /// <seealso cref="GrowIndividual(int, int, int, int)"/>
        /// <seealso cref="GrowSide(Side, int)"/>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown <see cref="Rect2i"/>.</returns>
        public Rect2i Grow(int by)
        {
            Rect2i g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2i"/> grown by the specified amount
        /// on each side individually.
        /// </summary>
        /// <seealso cref="Grow(int)"/>
        /// <seealso cref="GrowSide(Side, int)"/>
        /// <param name="left">The amount to grow by on the left side.</param>
        /// <param name="top">The amount to grow by on the top side.</param>
        /// <param name="right">The amount to grow by on the right side.</param>
        /// <param name="bottom">The amount to grow by on the bottom side.</param>
        /// <returns>The grown <see cref="Rect2i"/>.</returns>
        public Rect2i GrowIndividual(int left, int top, int right, int bottom)
        {
            Rect2i g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2i"/> grown by the specified amount
        /// on the specified <see cref="Side"/>.
        /// </summary>
        /// <seealso cref="Grow(int)"/>
        /// <seealso cref="GrowIndividual(int, int, int, int)"/>
        /// <param name="side">The side to grow.</param>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown <see cref="Rect2i"/>.</returns>
        public Rect2i GrowSide(Side side, int by)
        {
            Rect2i g = this;

            g = g.GrowIndividual(Side.Left == side ? by : 0,
                    Side.Top == side ? by : 0,
                    Side.Right == side ? by : 0,
                    Side.Bottom == side ? by : 0);

            return g;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rect2i"/> is flat or empty,
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="Rect2i"/> has area.
        /// </returns>
        public bool HasNoArea()
        {
            return _size.x <= 0 || _size.y <= 0;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rect2i"/> contains a point,
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="Rect2i"/> contains <paramref name="point"/>.
        /// </returns>
        public bool HasPoint(Vector2i point)
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
        /// Returns <see langword="true"/> if the <see cref="Rect2i"/> overlaps with <paramref name="b"/>
        /// (i.e. they have at least one point in common).
        ///
        /// If <paramref name="includeBorders"/> is <see langword="true"/>,
        /// they will also be considered overlapping if their borders touch,
        /// even without intersection.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2i"/> to check for intersections with.</param>
        /// <param name="includeBorders">Whether or not to consider borders.</param>
        /// <returns>A <see langword="bool"/> for whether or not they are intersecting.</returns>
        public bool Intersects(Rect2i b, bool includeBorders = false)
        {
            if (includeBorders)
            {
                if (_position.x > b._position.x + b._size.x)
                    return false;
                if (_position.x + _size.x < b._position.x)
                    return false;
                if (_position.y > b._position.y + b._size.y)
                    return false;
                if (_position.y + _size.y < b._position.y)
                    return false;
            }
            else
            {
                if (_position.x >= b._position.x + b._size.x)
                    return false;
                if (_position.x + _size.x <= b._position.x)
                    return false;
                if (_position.y >= b._position.y + b._size.y)
                    return false;
                if (_position.y + _size.y <= b._position.y)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Returns a larger <see cref="Rect2i"/> that contains this <see cref="Rect2i"/> and <paramref name="b"/>.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2i"/>.</param>
        /// <returns>The merged <see cref="Rect2i"/>.</returns>
        public Rect2i Merge(Rect2i b)
        {
            Rect2i newRect;

            newRect._position.x = Mathf.Min(b._position.x, _position.x);
            newRect._position.y = Mathf.Min(b._position.y, _position.y);

            newRect._size.x = Mathf.Max(b._position.x + b._size.x, _position.x + _size.x);
            newRect._size.y = Mathf.Max(b._position.y + b._size.y, _position.y + _size.y);

            newRect._size -= newRect._position; // Make relative again

            return newRect;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2i"/> from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size.</param>
        public Rect2i(Vector2i position, Vector2i size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2i"/> from a position, width, and height.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2i(Vector2i position, int width, int height)
        {
            _position = position;
            _size = new Vector2i(width, height);
        }

        /// <summary>
        /// Constructs a <see cref="Rect2i"/> from x, y, and size.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="size">The size.</param>
        public Rect2i(int x, int y, Vector2i size)
        {
            _position = new Vector2i(x, y);
            _size = size;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2i"/> from x, y, width, and height.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2i(int x, int y, int width, int height)
        {
            _position = new Vector2i(x, y);
            _size = new Vector2i(width, height);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2i"/>s are exactly equal.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are equal.</returns>
        public static bool operator ==(Rect2i left, Rect2i right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2i"/>s are not equal.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are not equal.</returns>
        public static bool operator !=(Rect2i left, Rect2i right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Converts this <see cref="Rect2i"/> to a <see cref="Rect2"/>.
        /// </summary>
        /// <param name="value">The rect to convert.</param>
        public static implicit operator Rect2(Rect2i value)
        {
            return new Rect2(value._position, value._size);
        }

        /// <summary>
        /// Converts a <see cref="Rect2"/> to a <see cref="Rect2i"/>.
        /// </summary>
        /// <param name="value">The rect to convert.</param>
        public static explicit operator Rect2i(Rect2 value)
        {
            return new Rect2i((Vector2i)value.Position, (Vector2i)value.Size);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this rect and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the rect and the other object are equal.</returns>
        public override bool Equals(object obj)
        {
            if (obj is Rect2i)
            {
                return Equals((Rect2i)obj);
            }

            return false;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this rect and <paramref name="other"/> are equal.
        /// </summary>
        /// <param name="other">The other rect to compare.</param>
        /// <returns>Whether or not the rects are equal.</returns>
        public bool Equals(Rect2i other)
        {
            return _position.Equals(other._position) && _size.Equals(other._size);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Rect2i"/>.
        /// </summary>
        /// <returns>A hash code for this rect.</returns>
        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        /// <summary>
        /// Converts this <see cref="Rect2i"/> to a string.
        /// </summary>
        /// <returns>A string representation of this rect.</returns>
        public override string ToString()
        {
            return $"{_position}, {_size}";
        }

        /// <summary>
        /// Converts this <see cref="Rect2i"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this rect.</returns>
        public string ToString(string format)
        {
            return $"{_position.ToString(format)}, {_size.ToString(format)}";
        }
    }
}
