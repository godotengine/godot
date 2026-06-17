using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct Transform2D : IEquatable<Transform2D>
	{
		public Vector2 X;
		public Vector2 Y;
		public Vector2 Origin;

		public readonly float Rotation => MathF.Atan2(X.Y, X.X);

		public readonly Vector2 Scale
		{
			get
			{
				float detSign = Sign(Determinant());
				return new Vector2(Length(X), detSign * Length(Y));
			}
		}

		public readonly float Skew
		{
			get
			{
				float detSign = Sign(Determinant());
				Vector2 nx = Normalize(X);
				Vector2 ny = Normalize(Mul(Y, detSign));
				float d = Clamp(Dot(nx, ny), -1f, 1f);
				return MathF.Acos(d) - MathF.PI * 0.5f;
			}
		}

		public Vector2 this[int column]
		{
			readonly get
			{
				return column switch
				{
					0 => X,
					1 => Y,
					2 => Origin,
					_ => throw new ArgumentOutOfRangeException(nameof(column))
				};
			}
			set
			{
				switch (column)
				{
					case 0:
						X = value;
						return;
					case 1:
						Y = value;
						return;
					case 2:
						Origin = value;
						return;
					default:
						throw new ArgumentOutOfRangeException(nameof(column));
				}
			}
		}

		public float this[int column, int row]
		{
			readonly get
			{
				Vector2 v = this[column];
				return row switch
				{
					0 => v.X,
					1 => v.Y,
					_ => throw new ArgumentOutOfRangeException(nameof(row))
				};
			}
			set
			{
				Vector2 v = this[column];
				switch (row)
				{
					case 0:
						v.X = value;
						break;
					case 1:
						v.Y = value;
						break;
					default:
						throw new ArgumentOutOfRangeException(nameof(row));
				}
				this[column] = v;
			}
		}

		public readonly Transform2D AffineInverse()
		{
			float det = Determinant();

			if (det == 0f)
				throw new InvalidOperationException("Matrix determinant is zero and cannot be inverted.");

			Transform2D inv = this;

			inv[0, 0] = this[1, 1];
			inv[1, 1] = this[0, 0];

			float detInv = 1.0f / det;

			inv[0] = Mul(inv[0], new Vector2(detInv, -detInv));
			inv[1] = Mul(inv[1], new Vector2(-detInv, detInv));

			inv[2] = inv.BasisXform(Neg(inv[2]));

			return inv;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly float Determinant()
		{
			return (X.X * Y.Y) - (X.Y * Y.X);
		}

		public readonly Vector2 BasisXform(Vector2 v)
		{
			return new Vector2(Tdotx(v), Tdoty(v));
		}

		public readonly Vector2 BasisXformInv(Vector2 v)
		{
			return new Vector2(Dot(X, v), Dot(Y, v));
		}

		public readonly Transform2D InterpolateWith(Transform2D transform, float weight)
		{
			return new Transform2D(
				LerpAngle(Rotation, transform.Rotation, weight),
				Lerp(Scale, transform.Scale, weight),
				LerpAngle(Skew, transform.Skew, weight),
				Lerp(Origin, transform.Origin, weight)
			);
		}

		public readonly Transform2D Inverse()
		{
			Transform2D inv = this;

			inv.X.Y = Y.X;
			inv.Y.X = X.Y;

			inv.Origin = inv.BasisXform(Neg(inv.Origin));

			return inv;
		}

		public readonly bool IsFinite()
		{
			return IsFinite(X) && IsFinite(Y) && IsFinite(Origin);
		}

		public readonly Transform2D Orthonormalized()
		{
			Transform2D ortho = this;

			Vector2 orthoX = Normalize(ortho.X);
			Vector2 orthoY = Sub(ortho.Y, Mul(orthoX, Dot(orthoX, ortho.Y)));
			orthoY = Normalize(orthoY);

			ortho.X = orthoX;
			ortho.Y = orthoY;

			return ortho;
		}

		public readonly Transform2D Rotated(float angle)
		{
			return new Transform2D(angle, new Vector2()) * this;
		}

		public readonly Transform2D RotatedLocal(float angle)
		{
			return this * new Transform2D(angle, new Vector2());
		}

		public readonly Transform2D Scaled(Vector2 scale)
		{
			Transform2D copy = this;
			copy.X = Mul(copy.X, scale);
			copy.Y = Mul(copy.Y, scale);
			copy.Origin = Mul(copy.Origin, scale);
			return copy;
		}

		public readonly Transform2D ScaledLocal(Vector2 scale)
		{
			Transform2D copy = this;
			copy.X = Mul(copy.X, scale);
			copy.Y = Mul(copy.Y, scale);
			return copy;
		}

		private readonly float Tdotx(Vector2 with)
		{
			return (this[0, 0] * with.X) + (this[1, 0] * with.Y);
		}

		private readonly float Tdoty(Vector2 with)
		{
			return (this[0, 1] * with.X) + (this[1, 1] * with.Y);
		}

		public readonly Transform2D Translated(Vector2 offset)
		{
			Transform2D copy = this;
			copy.Origin = Add(copy.Origin, offset);
			return copy;
		}

		public readonly Transform2D TranslatedLocal(Vector2 offset)
		{
			Transform2D copy = this;
			copy.Origin = Add(copy.Origin, copy.BasisXform(offset));
			return copy;
		}

		private static readonly Transform2D _identity = new Transform2D(1, 0, 0, 1, 0, 0);
		private static readonly Transform2D _flipX = new Transform2D(-1, 0, 0, 1, 0, 0);
		private static readonly Transform2D _flipY = new Transform2D(1, 0, 0, -1, 0, 0);

		public static Transform2D Identity => _identity;
		public static Transform2D FlipX => _flipX;
		public static Transform2D FlipY => _flipY;

		public Transform2D(Vector2 xAxis, Vector2 yAxis, Vector2 originPos)
		{
			X = xAxis;
			Y = yAxis;
			Origin = originPos;
		}

		public Transform2D(float xx, float xy, float yx, float yy, float ox, float oy)
		{
			X = new Vector2(xx, xy);
			Y = new Vector2(yx, yy);
			Origin = new Vector2(ox, oy);
		}

		public Transform2D(float rotation, Vector2 origin)
		{
			float sin = MathF.Sin(rotation);
			float cos = MathF.Cos(rotation);

			X = new Vector2(cos, sin);
			Y = new Vector2(-sin, cos);
			Origin = origin;
		}

		public Transform2D(float rotation, Vector2 scale, float skew, Vector2 origin)
		{
			float rotationSin = MathF.Sin(rotation);
			float rotationCos = MathF.Cos(rotation);
			float rotationSkewSin = MathF.Sin(rotation + skew);
			float rotationSkewCos = MathF.Cos(rotation + skew);

			X = new Vector2(rotationCos * scale.X, rotationSin * scale.X);
			Y = new Vector2(-rotationSkewSin * scale.Y, rotationSkewCos * scale.Y);
			Origin = origin;
		}

		public static Transform2D operator *(Transform2D left, Transform2D right)
		{
			left.Origin = left * right.Origin;

			float x0 = left.Tdotx(right.X);
			float x1 = left.Tdoty(right.X);
			float y0 = left.Tdotx(right.Y);
			float y1 = left.Tdoty(right.Y);

			left.X.X = x0;
			left.X.Y = x1;
			left.Y.X = y0;
			left.Y.Y = y1;

			return left;
		}

		public static Vector2 operator *(Transform2D transform, Vector2 vector)
		{
			return Add(new Vector2(transform.Tdotx(vector), transform.Tdoty(vector)), transform.Origin);
		}

		public static Vector2 operator *(Vector2 vector, Transform2D transform)
		{
			Vector2 vInv = Sub(vector, transform.Origin);
			return new Vector2(Dot(transform.X, vInv), Dot(transform.Y, vInv));
		}

		public static Rect2 operator *(Transform2D transform, Rect2 rect)
		{
			Vector2 pos = transform * rect.Position;
			Vector2 toX = Mul(transform.X, rect.Size.X);
			Vector2 toY = Mul(transform.Y, rect.Size.Y);

			return new Rect2(pos, new Vector2())
				.Expand(Add(pos, toX))
				.Expand(Add(pos, toY))
				.Expand(Add(Add(pos, toX), toY));
		}

		public static Rect2 operator *(Rect2 rect, Transform2D transform)
		{
			Vector2 pos = rect.Position * transform;
			Vector2 to1 = new Vector2(rect.Position.X, rect.Position.Y + rect.Size.Y) * transform;
			Vector2 to2 = new Vector2(rect.Position.X + rect.Size.X, rect.Position.Y + rect.Size.Y) * transform;
			Vector2 to3 = new Vector2(rect.Position.X + rect.Size.X, rect.Position.Y) * transform;

			return new Rect2(pos, new Vector2()).Expand(to1).Expand(to2).Expand(to3);
		}

		public static Vector2[] operator *(Transform2D transform, Vector2[] array)
		{
			Vector2[] newArray = new Vector2[array.Length];

			for (int i = 0; i < array.Length; i++)
			{
				newArray[i] = transform * array[i];
			}

			return newArray;
		}

		public static Vector2[] operator *(Vector2[] array, Transform2D transform)
		{
			Vector2[] newArray = new Vector2[array.Length];

			for (int i = 0; i < array.Length; i++)
			{
				newArray[i] = array[i] * transform;
			}

			return newArray;
		}

		public static bool operator ==(Transform2D left, Transform2D right)
		{
			return left.Equals(right);
		}

		public static bool operator !=(Transform2D left, Transform2D right)
		{
			return !left.Equals(right);
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Transform2D other && Equals(other);
		}

		public readonly bool Equals(Transform2D other)
		{
			return X.Equals(other.X) && Y.Equals(other.Y) && Origin.Equals(other.Origin);
		}

		public readonly bool IsEqualApprox(Transform2D other)
		{
			return IsEqualApprox(X, other.X) && IsEqualApprox(Y, other.Y) && IsEqualApprox(Origin, other.Origin);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(X, Y, Origin);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			return $"[X: {FormatVector(X, format)}, Y: {FormatVector(Y, format)}, O: {FormatVector(Origin, format)}]";
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

		private static Vector2 Mul(Vector2 a, Vector2 b)
		{
			return new Vector2(a.X * b.X, a.Y * b.Y);
		}

		private static Vector2 Neg(Vector2 v)
		{
			return new Vector2(-v.X, -v.Y);
		}

		private static float Dot(Vector2 a, Vector2 b)
		{
			return (a.X * b.X) + (a.Y * b.Y);
		}

		private static float Length(Vector2 v)
		{
			return MathF.Sqrt((v.X * v.X) + (v.Y * v.Y));
		}

		private static Vector2 Normalize(Vector2 v)
		{
			float len = Length(v);
			if (len == 0f)
				return new Vector2();

			return new Vector2(v.X / len, v.Y / len);
		}

		private static Vector2 Lerp(Vector2 from, Vector2 to, float weight)
		{
			return new Vector2(
				from.X + ((to.X - from.X) * weight),
				from.Y + ((to.Y - from.Y) * weight)
			);
		}

		private static float LerpAngle(float from, float to, float weight)
		{
			float delta = Repeat(to - from, MathF.PI * 2f);
			if (delta > MathF.PI)
				delta -= MathF.PI * 2f;

			return from + (delta * weight);
		}

		private static float Repeat(float t, float length)
		{
			return t - MathF.Floor(t / length) * length;
		}

		private static float Clamp(float value, float min, float max)
		{
			return value < min ? min : (value > max ? max : value);
		}

		private static float Sign(float value)
		{
			return value < 0f ? -1f : 1f;
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

		private static string FormatVector(Vector2 v, string? format)
		{
			return $"({v.X.ToString(format)}, {v.Y.ToString(format)})";
		}
	}
}
