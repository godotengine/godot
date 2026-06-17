using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

#nullable enable
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
			readonly get { return _position; }
			set { _position = value; }
		}

		public Vector2 Size
		{
			readonly get { return _size; }
			set { _size = value; }
		}

		public Vector2 End
		{
			readonly get { return Add(_position, _size); }
			set { _size = Sub(value, _position); }
		}

		public float Area
		{
			get { return _size.X * _size.Y; }
		}

		public readonly Rect2 Abs()
		{
			Vector2 end = End;
			Vector2 topLeft = Min(end, _position);
			return new Rect2(topLeft, Abs(_size));
		}

		public readonly Rect2 Intersection(Rect2 b)
		{
			Rect2 newRect = b;

			if (!Intersects(newRect))
			{
				return new Rect2();
			}

			newRect._position = Max(b._position, _position);

			Vector2 bEnd = Add(b._position, b._size);
			Vector2 end = Add(_position, _size);

			newRect._size = Sub(Min(bEnd, end), newRect._position);
			return newRect;
		}

		public readonly bool IsFinite()
		{
			return IsFinite(_position) && IsFinite(_size);
		}

		public readonly bool Encloses(Rect2 b)
		{
			return b._position.X >= _position.X &&
				   b._position.Y >= _position.Y &&
				   b._position.X + b._size.X <= _position.X + _size.X &&
				   b._position.Y + b._size.Y <= _position.Y + _size.Y;
		}

		public readonly Rect2 Expand(Vector2 to)
		{
			Rect2 expanded = this;

			Vector2 begin = expanded._position;
			Vector2 end = Add(expanded._position, expanded._size);

			if (to.X < begin.X) begin.X = to.X;
			if (to.Y < begin.Y) begin.Y = to.Y;

			if (to.X > end.X) end.X = to.X;
			if (to.Y > end.Y) end.Y = to.Y;

			expanded._position = begin;
			expanded._size = Sub(end, begin);
			return expanded;
		}

		public readonly Vector2 GetCenter()
		{
			return Add(_position, Mul(_size, 0.5f));
		}

		public readonly Vector2 GetSupport(Vector2 direction)
		{
			Vector2 support = _position;
			if (direction.X > 0.0f)
			{
				support.X += _size.X;
			}
			if (direction.Y > 0.0f)
			{
				support.Y += _size.Y;
			}
			return support;
		}

		public readonly Rect2 Grow(float by)
		{
			Rect2 g = this;

			g._position.X -= by;
			g._position.Y -= by;
			g._size.X += by * 2f;
			g._size.Y += by * 2f;

			return g;
		}

		public readonly Rect2 GrowIndividual(float left, float top, float right, float bottom)
		{
			Rect2 g = this;

			g._position.X -= left;
			g._position.Y -= top;
			g._size.X += left + right;
			g._size.Y += top + bottom;

			return g;
		}

		public readonly Rect2 GrowSide(Side side, float by)
		{
			return GrowIndividual(
				Side.SIDE_LEFT == side ? by : 0f,
				Side.SIDE_TOP == side ? by : 0f,
				Side.SIDE_RIGHT == side ? by : 0f,
				Side.SIDE_BOTTOM == side ? by : 0f
			);
		}

		public readonly bool HasArea()
		{
			return _size.X > 0.0f && _size.Y > 0.0f;
		}

		public readonly bool HasPoint(Vector2 point)
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

		public readonly bool Intersects(Rect2 b, bool includeBorders = false)
		{
			if (includeBorders)
			{
				if (_position.X > b._position.X + b._size.X) return false;
				if (_position.X + _size.X < b._position.X) return false;
				if (_position.Y > b._position.Y + b._size.Y) return false;
				if (_position.Y + _size.Y < b._position.Y) return false;
			}
			else
			{
				if (_position.X >= b._position.X + b._size.X) return false;
				if (_position.X + _size.X <= b._position.X) return false;
				if (_position.Y >= b._position.Y + b._size.Y) return false;
				if (_position.Y + _size.Y <= b._position.Y) return false;
			}

			return true;
		}

		public readonly Rect2 Merge(Rect2 b)
		{
			Rect2 newRect;

			newRect._position = Min(b._position, _position);
			Vector2 endA = Add(_position, _size);
			Vector2 endB = Add(b._position, b._size);
			newRect._size = Sub(Max(endA, endB), newRect._position);

			return newRect;
		}

		public Rect2(Vector2 position, Vector2 size)
		{
			_position = position;
			_size = size;
		}

		public Rect2(Vector2 position, float width, float height)
		{
			_position = position;
			_size = new Vector2(width, height);
		}

		public Rect2(float x, float y, Vector2 size)
		{
			_position = new Vector2(x, y);
			_size = size;
		}

		public Rect2(float x, float y, float width, float height)
		{
			_position = new Vector2(x, y);
			_size = new Vector2(width, height);
		}

		public static bool operator ==(Rect2 left, Rect2 right) => left.Equals(right);
		public static bool operator !=(Rect2 left, Rect2 right) => !left.Equals(right);

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Rect2 other && Equals(other);
		}

		public readonly bool Equals(Rect2 other)
		{
			return _position.Equals(other._position) && _size.Equals(other._size);
		}

		public readonly bool IsEqualApprox(Rect2 other)
		{
			return IsEqualApprox(_position, other._position) && IsEqualApprox(_size, other._size);
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

		private static Vector2 Add(Vector2 a, Vector2 b)
		{
			return new Vector2(a.X + b.X, a.Y + b.Y);
		}

		private static Vector2 Sub(Vector2 a, Vector2 b)
		{
			return new Vector2(a.X - b.X, a.Y - b.Y);
		}

		private static Vector2 Mul(Vector2 v, float s)
		{
			return new Vector2(v.X * s, v.Y * s);
		}

		private static Vector2 Min(Vector2 a, Vector2 b)
		{
			return new Vector2(
				a.X < b.X ? a.X : b.X,
				a.Y < b.Y ? a.Y : b.Y
			);
		}

		private static Vector2 Max(Vector2 a, Vector2 b)
		{
			return new Vector2(
				a.X > b.X ? a.X : b.X,
				a.Y > b.Y ? a.Y : b.Y
			);
		}

		private static Vector2 Abs(Vector2 v)
		{
			return new Vector2(MathF.Abs(v.X), MathF.Abs(v.Y));
		}

		private static bool IsFinite(Vector2 v)
		{
			return IsFinite(v.X) && IsFinite(v.Y);
		}

		private static bool IsFinite(float value)
		{
			return !float.IsNaN(value) && !float.IsInfinity(value);
		}

		private static bool IsEqualApprox(Vector2 a, Vector2 b)
		{
			return MathF.Abs(a.X - b.X) <= 0.00001f &&
				   MathF.Abs(a.Y - b.Y) <= 0.00001f;
		}
	}
}
