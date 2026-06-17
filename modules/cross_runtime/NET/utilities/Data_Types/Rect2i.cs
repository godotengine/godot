using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

#nullable enable
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
			readonly get { return _position; }
			set { _position = value; }
		}

		public Vector2i Size
		{
			readonly get { return _size; }
			set { _size = value; }
		}

		public Vector2i End
		{
			readonly get { return Add(_position, _size); }
			set { _size = Sub(value, _position); }
		}

		public readonly int Area
		{
			get { return _size.X * _size.Y; }
		}

		public readonly Rect2i Abs()
		{
			Vector2i end = End;
			Vector2i topLeft = Min(end, _position);
			return new Rect2i(topLeft, Abs(_size));
		}

		public readonly Rect2i Intersection(Rect2i b)
		{
			Rect2i newRect = b;

			if (!Intersects(newRect))
			{
				return new Rect2i();
			}

			newRect._position = Max(b._position, _position);

			Vector2i bEnd = Add(b._position, b._size);
			Vector2i end = Add(_position, _size);

			newRect._size = Sub(Min(bEnd, end), newRect._position);
			return newRect;
		}

		public readonly bool Encloses(Rect2i b)
		{
			return b._position.X >= _position.X && b._position.Y >= _position.Y &&
				   b._position.X + b._size.X <= _position.X + _size.X &&
				   b._position.Y + b._size.Y <= _position.Y + _size.Y;
		}

		public readonly Rect2i Expand(Vector2i to)
		{
			Rect2i expanded = this;

			Vector2i begin = expanded._position;
			Vector2i end = Add(expanded._position, expanded._size);

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
			expanded._size = Sub(end, begin);

			return expanded;
		}

		public readonly Vector2i GetCenter()
		{
			return Add(_position, Div(_size, 2));
		}

		public readonly Rect2i Grow(int by)
		{
			Rect2i g = this;

			g._position.X -= by;
			g._position.Y -= by;
			g._size.X += by * 2;
			g._size.Y += by * 2;

			return g;
		}

		public readonly Rect2i GrowIndividual(int left, int top, int right, int bottom)
		{
			Rect2i g = this;

			g._position.X -= left;
			g._position.Y -= top;
			g._size.X += left + right;
			g._size.Y += top + bottom;

			return g;
		}

		public readonly Rect2i GrowSide(Side side, int by)
		{
			return GrowIndividual(
				Side.SIDE_LEFT == side ? by : 0,
				Side.SIDE_TOP == side ? by : 0,
				Side.SIDE_RIGHT == side ? by : 0,
				Side.SIDE_BOTTOM == side ? by : 0
			);
		}

		public readonly bool HasArea()
		{
			return _size.X > 0 && _size.Y > 0;
		}

		public readonly bool HasPoint(Vector2i point)
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

		public readonly bool Intersects(Rect2i b)
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

		public readonly Rect2i Merge(Rect2i b)
		{
			Rect2i newRect;

			newRect._position = Min(b._position, _position);
			newRect._size = Max(Add(b._position, b._size), Add(_position, _size));
			newRect._size = Sub(newRect._size, newRect._position);

			return newRect;
		}

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
			return new Rect2(
				new Vector2(value._position.X, value._position.Y),
				new Vector2(value._size.X, value._size.Y)
			);
		}

		public static explicit operator Rect2i(Rect2 value)
		{
			return new Rect2i(
				new Vector2i((int)value.Position.X, (int)value.Position.Y),
				new Vector2i((int)value.Size.X, (int)value.Size.Y)
			);
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Rect2i other && Equals(other);
		}

		public readonly bool Equals(Rect2i other)
		{
			return _position.Equals(other._position) && _size.Equals(other._size);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(_position, _size);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			return $"{_position.ToString(format)}, {_size.ToString(format)}";
		}

		private static Vector2i Add(Vector2i a, Vector2i b)
		{
			return new Vector2i(a.X + b.X, a.Y + b.Y);
		}

		private static Vector2i Sub(Vector2i a, Vector2i b)
		{
			return new Vector2i(a.X - b.X, a.Y - b.Y);
		}

		private static Vector2i Mul(Vector2i v, int s)
		{
			return new Vector2i(v.X * s, v.Y * s);
		}

		private static Vector2i Div(Vector2i v, int s)
		{
			return new Vector2i(v.X / s, v.Y / s);
		}

		private static Vector2i Min(Vector2i a, Vector2i b)
		{
			return new Vector2i(
				a.X < b.X ? a.X : b.X,
				a.Y < b.Y ? a.Y : b.Y
			);
		}

		private static Vector2i Max(Vector2i a, Vector2i b)
		{
			return new Vector2i(
				a.X > b.X ? a.X : b.X,
				a.Y > b.Y ? a.Y : b.Y
			);
		}

		private static Vector2i Abs(Vector2i v)
		{
			return new Vector2i(Math.Abs(v.X), Math.Abs(v.Y));
		}
	}
}
