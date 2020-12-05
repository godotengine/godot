using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    /// <summary>
    /// 2D axis-aligned bounding box. Rect2 consists of a position, a size, and
    /// several utility functions. It is typically used for fast overlap tests.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2 : IEquatable<Rect2>
    {
        private Vector2 _position;
        private Vector2 _size;

        /// <summary>
        /// Beginning corner. Typically has values lower than End.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2 Position
        {
            get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from Position to End. Typically all components are positive.
        /// If the size is negative, you can use <see cref="Abs"/> to fix it.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2 Size
        {
            get { return _size; }
            set { _size = value; }
        }

        /// <summary>
        /// Ending corner. This is calculated as <see cref="Position"/> plus
        /// <see cref="Size"/>. Setting this value will change the size.
        /// </summary>
        /// <value>Getting is equivalent to `value = Position + Size`, setting is equivalent to `Size = value - Position`.</value>
        public Vector2 End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// The area of this rect.
        /// </summary>
        /// <value>Equivalent to <see cref="GetArea()"/>.</value>
        public real_t Area
        {
            get { return GetArea(); }
        }

        /// <summary>
        /// Returns a Rect2 with equivalent position and size, modified so that
        /// the top-left corner is the origin and width and height are positive.
        /// </summary>
        /// <returns>The modified rect.</returns>
        public Rect2 Abs()
        {
            Vector2 end = End;
            Vector2 topLeft = new Vector2(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns the intersection of this Rect2 and `b`.
        /// </summary>
        /// <param name="b">The other rect.</param>
        /// <returns>The clipped rect.</returns>
        public Rect2 Clip(Rect2 b)
        {
            var newRect = b;

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
        /// Returns true if this Rect2 completely encloses another one.
        /// </summary>
        /// <param name="b">The other rect that may be enclosed.</param>
        /// <returns>A bool for whether or not this rect encloses `b`.</returns>
        public bool Encloses(Rect2 b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        /// <summary>
        /// Returns this Rect2 expanded to include a given point.
        /// </summary>
        /// <param name="to">The point to include.</param>
        /// <returns>The expanded rect.</returns>
        public Rect2 Expand(Vector2 to)
        {
            var expanded = this;

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
        /// Returns the area of the Rect2.
        /// </summary>
        /// <returns>The area.</returns>
        public real_t GetArea()
        {
            return _size.x * _size.y;
        }

        /// <summary>
        /// Returns a copy of the Rect2 grown a given amount of units towards all the sides.
        /// </summary>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown rect.</returns>
        public Rect2 Grow(real_t by)
        {
            var g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        /// <summary>
        /// Returns a copy of the Rect2 grown a given amount of units towards each direction individually.
        /// </summary>
        /// <param name="left">The amount to grow by on the left.</param>
        /// <param name="top">The amount to grow by on the top.</param>
        /// <param name="right">The amount to grow by on the right.</param>
        /// <param name="bottom">The amount to grow by on the bottom.</param>
        /// <returns>The grown rect.</returns>
        public Rect2 GrowIndividual(real_t left, real_t top, real_t right, real_t bottom)
        {
            var g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        /// <summary>
        /// Returns a copy of the Rect2 grown a given amount of units towards the <see cref="Margin"/> direction.
        /// </summary>
        /// <param name="margin">The direction to grow in.</param>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown rect.</returns>
        public Rect2 GrowMargin(Margin margin, real_t by)
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
        /// Returns true if the Rect2 overlaps with `b`
        /// (i.e. they have at least one point in common).
        ///
        /// If `includeBorders` is true, they will also be considered overlapping
        /// if their borders touch, even without intersection.
        /// </summary>
        /// <param name="b">The other rect to check for intersections with.</param>
        /// <param name="includeBorders">Whether or not to consider borders.</param>
        /// <returns>A bool for whether or not they are intersecting.</returns>
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
        /// Returns a larger Rect2 that contains this Rect2 and `b`.
        /// </summary>
        /// <param name="b">The other rect.</param>
        /// <returns>The merged rect.</returns>
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
        /// Constructs a Rect2 from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size.</param>
        public Rect2(Vector2 position, Vector2 size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs a Rect2 from a position, width, and height.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2(Vector2 position, real_t width, real_t height)
        {
            _position = position;
            _size = new Vector2(width, height);
        }

        /// <summary>
        /// Constructs a Rect2 from x, y, and size.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="size">The size.</param>
        public Rect2(real_t x, real_t y, Vector2 size)
        {
            _position = new Vector2(x, y);
            _size = size;
        }

        /// <summary>
        /// Constructs a Rect2 from x, y, width, and height.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2(real_t x, real_t y, real_t width, real_t height)
        {
            _position = new Vector2(x, y);
            _size = new Vector2(width, height);
        }

        public static bool operator ==(Rect2 left, Rect2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Rect2 left, Rect2 right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Rect2)
            {
                return Equals((Rect2)obj);
            }

            return false;
        }

        public bool Equals(Rect2 other)
        {
            return _position.Equals(other._position) && _size.Equals(other._size);
        }

        /// <summary>
        /// Returns true if this rect and `other` are approximately equal, by running
        /// <see cref="Vector2.IsEqualApprox(Vector2)"/> on each component.
        /// </summary>
        /// <param name="other">The other rect to compare.</param>
        /// <returns>Whether or not the rects are approximately equal.</returns>
        public bool IsEqualApprox(Rect2 other)
        {
            return _position.IsEqualApprox(other._position) && _size.IsEqualApprox(other.Size);
        }

        public override int GetHashCode()
        {
            return _position.GetHashCode() ^ _size.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1})", new object[]
            {
                _position.ToString(),
                _size.ToString()
            });
        }

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
