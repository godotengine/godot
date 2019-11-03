using System;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

namespace Godot
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2 : IEquatable<Rect2>
    {
        private Vector2 _position;
        private Vector2 _size;

        public Vector2 Position
        {
            get { return _position; }
            set { _position = value; }
        }

        public Vector2 Size
        {
            get { return _size; }
            set { _size = value; }
        }

        public Vector2 End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        public real_t Area
        {
            get { return GetArea(); }
        }

        public Rect2 Abs()
        {
            Vector2 end = End;
            Vector2 topLeft = new Vector2(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2(topLeft, _size.Abs());
        }

        public Rect2 Clip(Rect2 b)
        {
            var newRect = b;

            if (!Intersects(newRect))
                return new Rect2();

            newRect._position.x = Mathf.Max(b._position.x, _position.x);
            newRect._position.y = Mathf.Max(b._position.y, _position.y);

            Vector2 bEnd = b._position + b._size;
            Vector2 end = _position + _size;

            newRect._size.x = Mathf.Min(bEnd.x, end.x) - newRect._position.x;
            newRect._size.y = Mathf.Min(bEnd.y, end.y) - newRect._position.y;

            return newRect;
        }

        public bool Encloses(Rect2 b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        public Rect2 Expand(Vector2 to)
        {
            var expanded = this;

            Vector2 begin = expanded._position;
            Vector2 end = expanded._position + expanded._size;

            if (to.x < begin.x)
                begin.x = to.x;
            if (to.y < begin.y)
                begin.y = to.y;

            if (to.x > end.x)
                end.x = to.x;
            if (to.y > end.y)
                end.y = to.y;

            expanded._position = begin;
            expanded._size = end - begin;

            return expanded;
        }

        public real_t GetArea()
        {
            return _size.x * _size.y;
        }

        public Rect2 Grow(real_t by)
        {
            var g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        public Rect2 GrowIndividual(real_t left, real_t top, real_t right, real_t bottom)
        {
            var g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        public Rect2 GrowMargin(Margin margin, real_t by)
        {
            var g = this;

            g.GrowIndividual(Margin.Left == margin ? by : 0,
                    Margin.Top == margin ? by : 0,
                    Margin.Right == margin ? by : 0,
                    Margin.Bottom == margin ? by : 0);

            return g;
        }

        public bool HasNoArea()
        {
            return _size.x <= 0 || _size.y <= 0;
        }

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

        public bool Intersects(Rect2 b)
        {
            if (_position.x >= b._position.x + b._size.x)
                return false;
            if (_position.x + _size.x <= b._position.x)
                return false;
            if (_position.y >= b._position.y + b._size.y)
                return false;
            if (_position.y + _size.y <= b._position.y)
                return false;

            return true;
        }

        public Rect2 Merge(Rect2 b)
        {
            Rect2 newRect;

            newRect._position.x = Mathf.Min(b._position.x, _position.x);
            newRect._position.y = Mathf.Min(b._position.y, _position.y);

            newRect._size.x = Mathf.Max(b._position.x + b._size.x, _position.x + _size.x);
            newRect._size.y = Mathf.Max(b._position.y + b._size.y, _position.y + _size.y);

            newRect._size = newRect._size - newRect._position; // Make relative again

            return newRect;
        }

        // Constructors
        public Rect2(Vector2 position, Vector2 size)
        {
            _position = position;
            _size = size;
        }
        public Rect2(Vector2 position, real_t width, real_t height)
        {
            _position = position;
            _size = new Vector2(width, height);
        }
        public Rect2(real_t x, real_t y, Vector2 size)
        {
            _position = new Vector2(x, y);
            _size = size;
        }
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
