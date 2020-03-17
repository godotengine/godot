using System;
using System.Runtime.InteropServices;

namespace Godot
{
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2i : IEquatable<Rect2i>
    {
        private Vector2i _position;
        private Vector2i _size;

        public Vector2i Position
        {
            get { return _position; }
            set { _position = value; }
        }

        public Vector2i Size
        {
            get { return _size; }
            set { _size = value; }
        }

        public Vector2i End
        {
            get { return _position + _size; }
            set { _size = value - _position; }
        }

        public int Area
        {
            get { return GetArea(); }
        }

        public Rect2i Abs()
        {
            Vector2i end = End;
            Vector2i topLeft = new Vector2i(Mathf.Min(_position.x, end.x), Mathf.Min(_position.y, end.y));
            return new Rect2i(topLeft, _size.Abs());
        }

        public Rect2i Clip(Rect2i b)
        {
            var newRect = b;

            if (!Intersects(newRect))
                return new Rect2i();

            newRect._position.x = Mathf.Max(b._position.x, _position.x);
            newRect._position.y = Mathf.Max(b._position.y, _position.y);

            Vector2i bEnd = b._position + b._size;
            Vector2i end = _position + _size;

            newRect._size.x = Mathf.Min(bEnd.x, end.x) - newRect._position.x;
            newRect._size.y = Mathf.Min(bEnd.y, end.y) - newRect._position.y;

            return newRect;
        }

        public bool Encloses(Rect2i b)
        {
            return b._position.x >= _position.x && b._position.y >= _position.y &&
               b._position.x + b._size.x < _position.x + _size.x &&
               b._position.y + b._size.y < _position.y + _size.y;
        }

        public Rect2i Expand(Vector2i to)
        {
            var expanded = this;

            Vector2i begin = expanded._position;
            Vector2i end = expanded._position + expanded._size;

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

        public int GetArea()
        {
            return _size.x * _size.y;
        }

        public Rect2i Grow(int by)
        {
            var g = this;

            g._position.x -= by;
            g._position.y -= by;
            g._size.x += by * 2;
            g._size.y += by * 2;

            return g;
        }

        public Rect2i GrowIndividual(int left, int top, int right, int bottom)
        {
            var g = this;

            g._position.x -= left;
            g._position.y -= top;
            g._size.x += left + right;
            g._size.y += top + bottom;

            return g;
        }

        public Rect2i GrowMargin(Margin margin, int by)
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

        public bool Intersects(Rect2i b)
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

        public Rect2i Merge(Rect2i b)
        {
            Rect2i newRect;

            newRect._position.x = Mathf.Min(b._position.x, _position.x);
            newRect._position.y = Mathf.Min(b._position.y, _position.y);

            newRect._size.x = Mathf.Max(b._position.x + b._size.x, _position.x + _size.x);
            newRect._size.y = Mathf.Max(b._position.y + b._size.y, _position.y + _size.y);

            newRect._size = newRect._size - newRect._position; // Make relative again

            return newRect;
        }

        // Constructors
        public Rect2i(Vector2i position, Vector2i size)
        {
            _position = position;
            _size = size;
        }
        public Rect2i(Vector2i position, int width, int height)
        {
            _position = position;
            _size = new Vector2i(width, height);
        }
        public Rect2i(int x, int y, Vector2i size)
        {
            _position = new Vector2i(x, y);
            _size = size;
        }
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
