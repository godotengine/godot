using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct Vector4 : IEquatable<Vector4>
	{
		public enum Axis
		{
			X = 0,
			Y,
			Z,
			W
		}

		public real_t X;
		public real_t Y;
		public real_t Z;
		public real_t W;

		public real_t this[int index]
		{
			readonly get
			{
				switch (index)
				{
					case 0:
						return X;
					case 1:
						return Y;
					case 2:
						return Z;
					case 3:
						return W;
					default:
						throw new ArgumentOutOfRangeException(nameof(index));
				}
			}
			set
			{
				switch (index)
				{
					case 0:
						X = value;
						return;
					case 1:
						Y = value;
						return;
					case 2:
						Z = value;
						return;
					case 3:
						W = value;
						return;
					default:
						throw new ArgumentOutOfRangeException(nameof(index));
				}
			}
		}

		public readonly void Deconstruct(out real_t x, out real_t y, out real_t z, out real_t w)
		{
			x = X;
			y = Y;
			z = Z;
			w = W;
		}

		internal void Normalize()
		{
			real_t lengthsq = LengthSquared();

			if (lengthsq == 0)
			{
				X = Y = Z = W = 0f;
			}
			else
			{
				real_t length = Mathf.Sqrt(lengthsq);
				X /= length;
				Y /= length;
				Z /= length;
				W /= length;
			}
		}

		public readonly Vector4 Abs()
		{
			return new Vector4(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z), Mathf.Abs(W));
		}

		public readonly Vector4 Ceil()
		{
			return new Vector4(Mathf.Ceil(X), Mathf.Ceil(Y), Mathf.Ceil(Z), Mathf.Ceil(W));
		}

		public readonly Vector4 Clamp(Vector4 min, Vector4 max)
		{
			return new Vector4
			(
				Mathf.Clamp(X, min.X, max.X),
				Mathf.Clamp(Y, min.Y, max.Y),
				Mathf.Clamp(Z, min.Z, max.Z),
				Mathf.Clamp(W, min.W, max.W)
			);
		}

		public readonly Vector4 Clamp(real_t min, real_t max)
		{
			return new Vector4
			(
				Mathf.Clamp(X, min, max),
				Mathf.Clamp(Y, min, max),
				Mathf.Clamp(Z, min, max),
				Mathf.Clamp(W, min, max)
			);
		}

		public readonly Vector4 CubicInterpolate(Vector4 b, Vector4 preA, Vector4 postB, real_t weight)
		{
			return new Vector4
			(
				Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight),
				Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight),
				Mathf.CubicInterpolate(Z, b.Z, preA.Z, postB.Z, weight),
				Mathf.CubicInterpolate(W, b.W, preA.W, postB.W, weight)
			);
		}

		public readonly Vector4 CubicInterpolateInTime(Vector4 b, Vector4 preA, Vector4 postB, real_t weight, real_t t, real_t preAT, real_t postBT)
		{
			return new Vector4
			(
				Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT),
				Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT),
				Mathf.CubicInterpolateInTime(Z, b.Z, preA.Z, postB.Z, weight, t, preAT, postBT),
				Mathf.CubicInterpolateInTime(W, b.W, preA.W, postB.W, weight, t, preAT, postBT)
			);
		}

		public readonly Vector4 DirectionTo(Vector4 to)
		{
			Vector4 ret = new Vector4(to.X - X, to.Y - Y, to.Z - Z, to.W - W);
			ret.Normalize();
			return ret;
		}

		public readonly real_t DistanceSquaredTo(Vector4 to)
		{
			return (to - this).LengthSquared();
		}

		public readonly real_t DistanceTo(Vector4 to)
		{
			return (to - this).Length();
		}

		public readonly real_t Dot(Vector4 with)
		{
			return (X * with.X) + (Y * with.Y) + (Z * with.Z) + (W * with.W);
		}

		public readonly Vector4 Floor()
		{
			return new Vector4(Mathf.Floor(X), Mathf.Floor(Y), Mathf.Floor(Z), Mathf.Floor(W));
		}

		public readonly Vector4 Inverse()
		{
			return new Vector4(1 / X, 1 / Y, 1 / Z, 1 / W);
		}

		public readonly bool IsFinite()
		{
			return Mathf.IsFinite(X) && Mathf.IsFinite(Y) && Mathf.IsFinite(Z) && Mathf.IsFinite(W);
		}

		public readonly bool IsNormalized()
		{
			return Mathf.IsEqualApprox(LengthSquared(), 1, Mathf.Epsilon);
		}

		public readonly real_t Length()
		{
			real_t x2 = X * X;
			real_t y2 = Y * Y;
			real_t z2 = Z * Z;
			real_t w2 = W * W;

			return Mathf.Sqrt(x2 + y2 + z2 + w2);
		}

		public readonly real_t LengthSquared()
		{
			real_t x2 = X * X;
			real_t y2 = Y * Y;
			real_t z2 = Z * Z;
			real_t w2 = W * W;

			return x2 + y2 + z2 + w2;
		}

		public readonly Vector4 Lerp(Vector4 to, real_t weight)
		{
			return new Vector4
			(
				Mathf.Lerp(X, to.X, weight),
				Mathf.Lerp(Y, to.Y, weight),
				Mathf.Lerp(Z, to.Z, weight),
				Mathf.Lerp(W, to.W, weight)
			);
		}

		public readonly Vector4 Max(Vector4 with)
		{
			return new Vector4
			(
				Mathf.Max(X, with.X),
				Mathf.Max(Y, with.Y),
				Mathf.Max(Z, with.Z),
				Mathf.Max(W, with.W)
			);
		}

		public readonly Vector4 Max(real_t with)
		{
			return new Vector4
			(
				Mathf.Max(X, with),
				Mathf.Max(Y, with),
				Mathf.Max(Z, with),
				Mathf.Max(W, with)
			);
		}

		public readonly Vector4 Min(Vector4 with)
		{
			return new Vector4
			(
				Mathf.Min(X, with.X),
				Mathf.Min(Y, with.Y),
				Mathf.Min(Z, with.Z),
				Mathf.Min(W, with.W)
			);
		}

		public readonly Vector4 Min(real_t with)
		{
			return new Vector4
			(
				Mathf.Min(X, with),
				Mathf.Min(Y, with),
				Mathf.Min(Z, with),
				Mathf.Min(W, with)
			);
		}

		public readonly Axis MaxAxisIndex()
		{
			int maxIndex = 0;
			real_t maxValue = X;

			for (int i = 1; i < 4; i++)
			{
				if (this[i] > maxValue)
				{
					maxIndex = i;
					maxValue = this[i];
				}
			}

			return (Axis)maxIndex;
		}

		public readonly Axis MinAxisIndex()
		{
			int minIndex = 0;
			real_t minValue = X;

			for (int i = 1; i < 4; i++)
			{
				if (this[i] <= minValue)
				{
					minIndex = i;
					minValue = this[i];
				}
			}

			return (Axis)minIndex;
		}

		public readonly Vector4 Normalized()
		{
			Vector4 v = this;
			v.Normalize();
			return v;
		}

		public readonly Vector4 PosMod(real_t mod)
		{
			return new Vector4
			(
				Mathf.PosMod(X, mod),
				Mathf.PosMod(Y, mod),
				Mathf.PosMod(Z, mod),
				Mathf.PosMod(W, mod)
			);
		}

		public readonly Vector4 PosMod(Vector4 modv)
		{
			return new Vector4
			(
				Mathf.PosMod(X, modv.X),
				Mathf.PosMod(Y, modv.Y),
				Mathf.PosMod(Z, modv.Z),
				Mathf.PosMod(W, modv.W)
			);
		}

		public readonly Vector4 Round()
		{
			return new Vector4(Mathf.Round(X), Mathf.Round(Y), Mathf.Round(Z), Mathf.Round(W));
		}

		public readonly Vector4 Sign()
		{
			Vector4 v;
			v.X = Mathf.Sign(X);
			v.Y = Mathf.Sign(Y);
			v.Z = Mathf.Sign(Z);
			v.W = Mathf.Sign(W);
			return v;
		}

		public readonly Vector4 Snapped(Vector4 step)
		{
			return new Vector4
			(
				Mathf.Snapped(X, step.X),
				Mathf.Snapped(Y, step.Y),
				Mathf.Snapped(Z, step.Z),
				Mathf.Snapped(W, step.W)
			);
		}

		public readonly Vector4 Snapped(real_t step)
		{
			return new Vector4
			(
				Mathf.Snapped(X, step),
				Mathf.Snapped(Y, step),
				Mathf.Snapped(Z, step),
				Mathf.Snapped(W, step)
			);
		}

		private static readonly Vector4 _zero = new Vector4(0, 0, 0, 0);
		private static readonly Vector4 _one = new Vector4(1, 1, 1, 1);
		private static readonly Vector4 _inf = new Vector4(Mathf.Inf, Mathf.Inf, Mathf.Inf, Mathf.Inf);

		public static Vector4 Zero { get { return _zero; } }
		public static Vector4 One { get { return _one; } }
		public static Vector4 Inf { get { return _inf; } }

		public Vector4(real_t x, real_t y, real_t z, real_t w)
		{
			X = x;
			Y = y;
			Z = z;
			W = w;
		}

		public static Vector4 operator +(Vector4 left, Vector4 right)
		{
			left.X += right.X;
			left.Y += right.Y;
			left.Z += right.Z;
			left.W += right.W;
			return left;
		}

		public static Vector4 operator -(Vector4 left, Vector4 right)
		{
			left.X -= right.X;
			left.Y -= right.Y;
			left.Z -= right.Z;
			left.W -= right.W;
			return left;
		}

		public static Vector4 operator -(Vector4 vec)
		{
			vec.X = -vec.X;
			vec.Y = -vec.Y;
			vec.Z = -vec.Z;
			vec.W = -vec.W;
			return vec;
		}

		public static Vector4 operator *(Vector4 vec, real_t scale)
		{
			vec.X *= scale;
			vec.Y *= scale;
			vec.Z *= scale;
			vec.W *= scale;
			return vec;
		}

		public static Vector4 operator *(real_t scale, Vector4 vec)
		{
			vec.X *= scale;
			vec.Y *= scale;
			vec.Z *= scale;
			vec.W *= scale;
			return vec;
		}

		public static Vector4 operator *(Vector4 left, Vector4 right)
		{
			left.X *= right.X;
			left.Y *= right.Y;
			left.Z *= right.Z;
			left.W *= right.W;
			return left;
		}

		public static Vector4 operator /(Vector4 vec, real_t divisor)
		{
			vec.X /= divisor;
			vec.Y /= divisor;
			vec.Z /= divisor;
			vec.W /= divisor;
			return vec;
		}

		public static Vector4 operator /(Vector4 vec, Vector4 divisorv)
		{
			vec.X /= divisorv.X;
			vec.Y /= divisorv.Y;
			vec.Z /= divisorv.Z;
			vec.W /= divisorv.W;
			return vec;
		}

		public static Vector4 operator %(Vector4 vec, real_t divisor)
		{
			vec.X %= divisor;
			vec.Y %= divisor;
			vec.Z %= divisor;
			vec.W %= divisor;
			return vec;
		}

		public static Vector4 operator %(Vector4 vec, Vector4 divisorv)
		{
			vec.X %= divisorv.X;
			vec.Y %= divisorv.Y;
			vec.Z %= divisorv.Z;
			vec.W %= divisorv.W;
			return vec;
		}

		public static bool operator ==(Vector4 left, Vector4 right)
		{
			return left.Equals(right);
		}

		public static bool operator !=(Vector4 left, Vector4 right)
		{
			return !left.Equals(right);
		}

		public static bool operator <(Vector4 left, Vector4 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					if (left.Z == right.Z)
					{
						return left.W < right.W;
					}
					return left.Z < right.Z;
				}
				return left.Y < right.Y;
			}
			return left.X < right.X;
		}

		public static bool operator >(Vector4 left, Vector4 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					if (left.Z == right.Z)
					{
						return left.W > right.W;
					}
					return left.Z > right.Z;
				}
				return left.Y > right.Y;
			}
			return left.X > right.X;
		}

		public static bool operator <=(Vector4 left, Vector4 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					if (left.Z == right.Z)
					{
						return left.W <= right.W;
					}
					return left.Z < right.Z;
				}
				return left.Y < right.Y;
			}
			return left.X < right.X;
		}

		public static bool operator >=(Vector4 left, Vector4 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					if (left.Z == right.Z)
					{
						return left.W >= right.W;
					}
					return left.Z > right.Z;
				}
				return left.Y > right.Y;
			}
			return left.X > right.X;
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Vector4 other && Equals(other);
		}

		public readonly bool Equals(Vector4 other)
		{
			return X == other.X && Y == other.Y && Z == other.Z && W == other.W;
		}

		public readonly bool IsEqualApprox(Vector4 other)
		{
			return Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y) && Mathf.IsEqualApprox(Z, other.Z) && Mathf.IsEqualApprox(W, other.W);
		}

		public readonly bool IsZeroApprox()
		{
			return Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y) && Mathf.IsZeroApprox(Z) && Mathf.IsZeroApprox(W);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(X, Y, Z, W);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)}, {W.ToString(format, CultureInfo.InvariantCulture)})";
		}
	}
}
