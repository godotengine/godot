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
        /// Beginning corner. Typically has values lower than End.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2i Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from Position to End. Typically all components are positive.
        /// If the size is negative, you can use <see cref="Abs"/> to fix it.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2i Size
        {
            get { return _size; }
            set { _size = value; }
        }

        /// <summary>
        /// Ending corner. This is calculated as <see cref="Position"/> plus
        /// <see cref="Size"/>. Setting this value will change the size.
        /// </summary>
        /// <value>Getting is equivalent to `value = Position + Size`, setting is equivalent to `Size = value - Position`.</value>
        public Vector2i End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// The area of this rect.
        /// </summary>
        /// <value>Equivalent to <see cref="GetArea()"/>.</value>
        public int Area
        {
            get { return GetArea(); }
        }

        /// <summary>
        /// Returns a Rect2i with equivalent position and size, modified so that
        /// the top-left corner is the origin and width and height are positive.
        /// </summary>
        /// <returns>The modified rect.</returns>
        public Rect2i Abs()
        {
            Vector2i end = End;
            Vector2i topLeft = new Vector2i(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2i(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns the intersection of this Rect2i and `b`.
        /// </summary>
        /// <param name="b">The other rect.</param>
        /// <returns>The clipped rect.</returns>
        public Rect2i Clip(Rect2i b)
        {
            var newRect = b;

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
        /// Returns true if this Rect2i completely encloses another one.
        /// </summary>
        /// <param name="b">The other rect that may be enclosed.</param>
        /// <returns>A bool for whether or not this rect encloses `b`.</returns>
        public bool Encloses(Rect2i b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        /// <summary>
        /// Returns this Rect2i expanded to include a given point.
        /// </summary>
        /// <param name="to">The point to include.</param>
        /// <returns>The expanded rect.</returns>
        public Rect2i Expand(Vector2i to)
        {
            var expanded = this;

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
        /// Returns the area of the Rect2.
        /// </summary>
        /// <returns>The area.</returns>
        public int GetArea()
        {
            return _size.x * _size.y;
        }

        /// <summary>
        /// Returns a copy of the Rect2i grown a given amount of units towards all the sides.
        /// </summary>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown rect.</returns>
        public Rect2i Grow(int by)
        {
            var g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        /// <summary>
        /// Returns a copy of the Rect2i grown a given amount of units towards each direction individually.
        /// </summary>
        /// <param name="left">The amount to grow by on the left.</param>
        /// <param name="top">The amount to grow by on the top.</param>
        /// <param name="right">The amount to grow by on the right.</param>
        /// <param name="bottom">The amount to grow by on the bottom.</param>
        /// <returns>The grown rect.</returns>
        public Rect2i GrowIndividual(int left, int top, int right, int bottom)
        {
            var g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        /// <summary>
        /// Returns a copy of the Rect2i grown a given amount of units towards the <see cref="Margin"/> direction.
        /// </summary>
        /// <param name="margin">The direction to grow in.</param>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown rect.</returns>
        public Rect2i GrowMargin(Margin margin, int by)
        {
            var g = this;

            g = g.GrowIndividual(Margin.Left == margin ? by : 0,
                    Margin.Top == margin ? by : 0,
                    Margin.Right == margin ? by : 0,
                    Margin.Bottom == margin ? by : 0);

            return g;
        }

        /// <summary>
        /// Returns true if the Rect2 is flat or empty, or false otherwise.
        /// </summary>
        /// <returns>A bool for whether or not the rect has area.</returns>
        public bool HasNoArea()
        {
            return _size.x <= 0 || _size.y <= 0;
        }

        /// <summary>
        /// Returns true if the Rect2 contains a point, or false otherwise.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>A bool for whether or not the rect contains `point`.</returns>
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
        /// Returns true if the Rect2i overlaps with `b`
        /// (i.e. they have at least one point in common).
        ///
        /// If `includeBorders` is true, they will also be considered overlapping
        /// if their borders touch, even without intersection.
        /// </summary>
        /// <param name="b">The other rect to check for intersections with.</param>
        /// <param name="includeBorders">Whether or not to consider borders.</param>
        /// <returns>A bool for whether or not they are intersecting.</returns>
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
        /// Returns a larger Rect2i that contains this Rect2 and `b`.
        /// </summary>
        /// <param name="b">The other rect.</param>
        /// <returns>The merged rect.</returns>
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
        /// Constructs a Rect2i from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size.</param>
        public Rect2i(Vector2i position, Vector2i size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs a Rect2i from a position, width, and height.
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
        /// Constructs a Rect2i from x, y, and size.
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
        /// Constructs a Rect2i from x, y, width, and height.
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

        public static bool operator ==(Rect2i left, Rect2i right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Rect2i left, Rect2i right)
        {
            return !left.Equals(right);
        }

        public static implicit operator Rect2(Rect2i value)
        {
            return new Rect2(value._position, value._size);
        }

        public static explicit operator Rect2i(Rect2 value)
        {
            return new Rect2i((Vector2i)value.Position, (Vector2i)value.Size);
        }

        public override bool Equals(object obj)
        {
            if (obj is Rect2i)
            {
                return Equals((Rect2i)obj);
            }

            return false;
        }

        public bool Equals(Rect2i other)
        {
            return _position.Equals(other._position) && _size.Equals(other._size);
        }

        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("{0}, {1}", new object[]
            {
                _position.ToString(),
                _size.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("{0}, {1}", new object[]
            {
                _position.ToString(format),
                _size.ToString(format)
            });
        }
    }
}
