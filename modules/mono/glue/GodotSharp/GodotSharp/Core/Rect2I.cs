using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

#nullable enable

namespace Godot
{
    /// <summary>
    /// 2D axis-aligned bounding box using integers. Rect2I consists of a position, a size, and
    /// several utility functions. It is typically used for fast overlap tests.
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Rect2I : IEquatable<Rect2I>
    {
        private Vector2I _position;
        private Vector2I _size;

        /// <summary>
        /// Beginning corner. Typically has values lower than <see cref="End"/>.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2I Position
        {
            readonly get { return _position; }
            set { _position = value; }
        }

        /// <summary>
        /// Size from <see cref="Position"/> to <see cref="End"/>. Typically all components are positive.
        /// If the size is negative, you can use <see cref="Abs"/> to fix it.
        /// </summary>
        /// <value>Directly uses a private field.</value>
        public Vector2I Size
        {
            readonly get { return _size; }
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
        public Vector2I End
        {
            readonly get { return _position + _size; }
            set { _size = value - _position; }
        }

        /// <summary>
        /// The area of this <see cref="Rect2I"/>.
        /// See also <see cref="HasArea"/>.
        /// </summary>
        public readonly int Area
        {
            get { return _size.X * _size.Y; }
        }

        /// <summary>
        /// Returns a <see cref="Rect2I"/> with equivalent position and size, modified so that
        /// the top-left corner is the origin and width and height are positive.
        /// </summary>
        /// <returns>The modified <see cref="Rect2I"/>.</returns>
        public readonly Rect2I Abs()
        {
            Vector2I end = End;
            Vector2I topLeft = end.Min(_position);
            return new Rect2I(topLeft, _size.Abs());
        }

        /// <summary>
        /// Returns the intersection of this <see cref="Rect2I"/> and <paramref name="b"/>.
        /// If the rectangles do not intersect, an empty <see cref="Rect2I"/> is returned.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2I"/>.</param>
        /// <returns>
        /// The intersection of this <see cref="Rect2I"/> and <paramref name="b"/>,
        /// or an empty <see cref="Rect2I"/> if they do not intersect.
        /// </returns>
        public readonly Rect2I Intersection(Rect2I b)
        {
            Rect2I newRect = b;

            if (!Intersects(newRect))
            {
                return new Rect2I();
            }

            newRect._position = b._position.Max(_position);

            Vector2I bEnd = b._position + b._size;
            Vector2I end = _position + _size;

            newRect._size = bEnd.Min(end) - newRect._position;

            return newRect;
        }

        /// <summary>
        /// Returns <see langword="true"/> if this <see cref="Rect2I"/> completely encloses another one.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2I"/> that may be enclosed.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not this <see cref="Rect2I"/> encloses <paramref name="b"/>.
        /// </returns>
        public readonly bool Encloses(Rect2I b)
        {
            return b._position.X >= _position.X && b._position.Y >= _position.Y &&
               b._position.X + b._size.X <= _position.X + _size.X &&
               b._position.Y + b._size.Y <= _position.Y + _size.Y;
        }

        /// <summary>
        /// Returns this <see cref="Rect2I"/> expanded to include a given point.
        /// </summary>
        /// <param name="to">The point to include.</param>
        /// <returns>The expanded <see cref="Rect2I"/>.</returns>
        public readonly Rect2I Expand(Vector2I to)
        {
            Rect2I expanded = this;

            Vector2I begin = expanded._position;
            Vector2I end = expanded._position + expanded._size;

            if (to.X < begin.X)
            {
                begin.X = to.X;
            }
            if (to.Y < begin.Y)
            {
                begin.Y = to.Y;
            }

            if (to.X > end.X)
            {
                end.X = to.X;
            }
            if (to.Y > end.Y)
            {
                end.Y = to.Y;
            }

            expanded._position = begin;
            expanded._size = end - begin;

            return expanded;
        }

        /// <summary>
        /// Returns the center of the <see cref="Rect2I"/>, which is equal
        /// to <see cref="Position"/> + (<see cref="Size"/> / 2).
        /// If <see cref="Size"/> is an odd number, the returned center
        /// value will be rounded towards <see cref="Position"/>.
        /// </summary>
        /// <returns>The center.</returns>
        public readonly Vector2I GetCenter()
        {
            return _position + (_size / 2);
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2I"/> grown by the specified amount
        /// on all sides.
        /// </summary>
        /// <seealso cref="GrowIndividual(int, int, int, int)"/>
        /// <seealso cref="GrowSide(Side, int)"/>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown <see cref="Rect2I"/>.</returns>
        public readonly Rect2I Grow(int by)
        {
            Rect2I g = this;

            g._position.X -= by;
            g._position.Y -= by;
            g._size.X += by * 2;
            g._size.Y += by * 2;

            return g;
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2I"/> grown by the specified amount
        /// on each side individually.
        /// </summary>
        /// <seealso cref="Grow(int)"/>
        /// <seealso cref="GrowSide(Side, int)"/>
        /// <param name="left">The amount to grow by on the left side.</param>
        /// <param name="top">The amount to grow by on the top side.</param>
        /// <param name="right">The amount to grow by on the right side.</param>
        /// <param name="bottom">The amount to grow by on the bottom side.</param>
        /// <returns>The grown <see cref="Rect2I"/>.</returns>
        public readonly Rect2I GrowIndividual(int left, int top, int right, int bottom)
        {
            Rect2I g = this;

            g._position.X -= left;
            g._position.Y -= top;
            g._size.X += left + right;
            g._size.Y += top + bottom;

            return g;
        }

        /// <summary>
        /// Returns a copy of the <see cref="Rect2I"/> grown by the specified amount
        /// on the specified <see cref="Side"/>.
        /// </summary>
        /// <seealso cref="Grow(int)"/>
        /// <seealso cref="GrowIndividual(int, int, int, int)"/>
        /// <param name="side">The side to grow.</param>
        /// <param name="by">The amount to grow by.</param>
        /// <returns>The grown <see cref="Rect2I"/>.</returns>
        public readonly Rect2I GrowSide(Side side, int by)
        {
            Rect2I g = this;

            g = g.GrowIndividual(Side.Left == side ? by : 0,
                    Side.Top == side ? by : 0,
                    Side.Right == side ? by : 0,
                    Side.Bottom == side ? by : 0);

            return g;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rect2I"/> has
        /// area, and <see langword="false"/> if the <see cref="Rect2I"/>
        /// is linear, empty, or has a negative <see cref="Size"/>.
        /// See also <see cref="Area"/>.
        /// </summary>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="Rect2I"/> has area.
        /// </returns>
        public readonly bool HasArea()
        {
            return _size.X > 0 && _size.Y > 0;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rect2I"/> contains a point,
        /// or <see langword="false"/> otherwise.
        /// </summary>
        /// <param name="point">The point to check.</param>
        /// <returns>
        /// A <see langword="bool"/> for whether or not the <see cref="Rect2I"/> contains <paramref name="point"/>.
        /// </returns>
        public readonly bool HasPoint(Vector2I point)
        {
            if (point.X < _position.X)
                return false;
            if (point.Y < _position.Y)
                return false;

            if (point.X >= _position.X + _size.X)
                return false;
            if (point.Y >= _position.Y + _size.Y)
                return false;

            return true;
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Rect2I"/> overlaps with <paramref name="b"/>
        /// (i.e. they have at least one point in common).
        /// </summary>
        /// <param name="b">The other <see cref="Rect2I"/> to check for intersections with.</param>
        /// <returns>A <see langword="bool"/> for whether or not they are intersecting.</returns>
        public readonly bool Intersects(Rect2I b)
        {
            if (_position.X >= b._position.X + b._size.X)
                return false;
            if (_position.X + _size.X <= b._position.X)
                return false;
            if (_position.Y >= b._position.Y + b._size.Y)
                return false;
            if (_position.Y + _size.Y <= b._position.Y)
                return false;

            return true;
        }

        /// <summary>
        /// Returns a larger <see cref="Rect2I"/> that contains this <see cref="Rect2I"/> and <paramref name="b"/>.
        /// </summary>
        /// <param name="b">The other <see cref="Rect2I"/>.</param>
        /// <returns>The merged <see cref="Rect2I"/>.</returns>
        public readonly Rect2I Merge(Rect2I b)
        {
            Rect2I newRect;

            newRect._position = b._position.Min(_position);

            newRect._size = (b._position + b._size).Max(_position + _size);

            newRect._size -= newRect._position; // Make relative again

            return newRect;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2I"/> from a position and size.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="size">The size.</param>
        public Rect2I(Vector2I position, Vector2I size)
        {
            _position = position;
            _size = size;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2I"/> from a position, width, and height.
        /// </summary>
        /// <param name="position">The position.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2I(Vector2I position, int width, int height)
        {
            _position = position;
            _size = new Vector2I(width, height);
        }

        /// <summary>
        /// Constructs a <see cref="Rect2I"/> from x, y, and size.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="size">The size.</param>
        public Rect2I(int x, int y, Vector2I size)
        {
            _position = new Vector2I(x, y);
            _size = size;
        }

        /// <summary>
        /// Constructs a <see cref="Rect2I"/> from x, y, width, and height.
        /// </summary>
        /// <param name="x">The position's X coordinate.</param>
        /// <param name="y">The position's Y coordinate.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        public Rect2I(int x, int y, int width, int height)
        {
            _position = new Vector2I(x, y);
            _size = new Vector2I(width, height);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2I"/>s are exactly equal.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are equal.</returns>
        public static bool operator ==(Rect2I left, Rect2I right)
        {
            return left.Equals(right);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the
        /// <see cref="Rect2I"/>s are not equal.
        /// </summary>
        /// <param name="left">The left rect.</param>
        /// <param name="right">The right rect.</param>
        /// <returns>Whether or not the rects are not equal.</returns>
        public static bool operator !=(Rect2I left, Rect2I right)
        {
            return !left.Equals(right);
        }

        /// <summary>
        /// Converts this <see cref="Rect2I"/> to a <see cref="Rect2"/>.
        /// </summary>
        /// <param name="value">The rect to convert.</param>
        public static implicit operator Rect2(Rect2I value)
        {
            return new Rect2(value._position, value._size);
        }

        /// <summary>
        /// Converts a <see cref="Rect2"/> to a <see cref="Rect2I"/>.
        /// </summary>
        /// <param name="value">The rect to convert.</param>
        public static explicit operator Rect2I(Rect2 value)
        {
            return new Rect2I((Vector2I)value.Position, (Vector2I)value.Size);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this rect and <paramref name="obj"/> are equal.
        /// </summary>
        /// <param name="obj">The other object to compare.</param>
        /// <returns>Whether or not the rect and the other object are equal.</returns>
        public override readonly bool Equals([NotNullWhen(true)] object? obj)
        {
            return obj is Rect2I other && Equals(other);
        }

        /// <summary>
        /// Returns <see langword="true"/> if this rect and <paramref name="other"/> are equal.
        /// </summary>
        /// <param name="other">The other rect to compare.</param>
        /// <returns>Whether or not the rects are equal.</returns>
        public readonly bool Equals(Rect2I other)
        {
            return _position.Equals(other._position) && _size.Equals(other._size);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="Rect2I"/>.
        /// </summary>
        /// <returns>A hash code for this rect.</returns>
        public override readonly int GetHashCode()
        {
            return HashCode.Combine(_position, _size);
        }

        /// <summary>
        /// Converts this <see cref="Rect2I"/> to a string.
        /// </summary>
        /// <returns>A string representation of this rect.</returns>
        public override readonly string ToString() => ToString(null);

        /// <summary>
        /// Converts this <see cref="Rect2I"/> to a string with the given <paramref name="format"/>.
        /// </summary>
        /// <returns>A string representation of this rect.</returns>
        public readonly string ToString(string? format)
        {
            return $"{_position.ToString(format)}, {_size.ToString(format)}";
        }
    }
}
