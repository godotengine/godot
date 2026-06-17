using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct Vector3 : IEquatable<Vector3>
	{
		public enum Axis
		{
			X = 0,
			Y,
			Z
		}

		public real_t X;
		public real_t Y;
		public real_t Z;

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
					default:
						throw new ArgumentOutOfRangeException(nameof(index));
				}
			}
		}

		public readonly void Deconstruct(out real_t x, out real_t y, out real_t z)
		{
			x = X;
			y = Y;
			z = Z;
		}

		internal void Normalize()
		{
			real_t lengthsq = LengthSquared();

			if (lengthsq == 0)
			{
				X = Y = Z = 0f;
			}
			else
			{
				real_t length = Mathf.Sqrt(lengthsq);
				X /= length;
				Y /= length;
				Z /= length;
			}
		}

		public readonly Vector3 Abs()
		{
			return new Vector3(Mathf.Abs(X), Mathf.Abs(Y), Mathf.Abs(Z));
		}

		public readonly real_t AngleTo(Vector3 to)
		{
			return Mathf.Atan2(Cross(to).Length(), Dot(to));
		}

		public readonly Vector3 Bounce(Vector3 normal)
		{
			return -Reflect(normal);
		}

		public readonly Vector3 Ceil()
		{
			return new Vector3(Mathf.Ceil(X), Mathf.Ceil(Y), Mathf.Ceil(Z));
		}

		public readonly Vector3 Clamp(Vector3 min, Vector3 max)
		{
			return new Vector3
			(
				Mathf.Clamp(X, min.X, max.X),
				Mathf.Clamp(Y, min.Y, max.Y),
				Mathf.Clamp(Z, min.Z, max.Z)
			);
		}

		public readonly Vector3 Clamp(real_t min, real_t max)
		{
			return new Vector3
			(
				Mathf.Clamp(X, min, max),
				Mathf.Clamp(Y, min, max),
				Mathf.Clamp(Z, min, max)
			);
		}

		public readonly Vector3 Cross(Vector3 with)
		{
			return new Vector3
			(
				(Y * with.Z) - (Z * with.Y),
				(Z * with.X) - (X * with.Z),
				(X * with.Y) - (Y * with.X)
			);
		}

		public readonly Vector3 CubicInterpolate(Vector3 b, Vector3 preA, Vector3 postB, real_t weight)
		{
			return new Vector3
			(
				Mathf.CubicInterpolate(X, b.X, preA.X, postB.X, weight),
				Mathf.CubicInterpolate(Y, b.Y, preA.Y, postB.Y, weight),
				Mathf.CubicInterpolate(Z, b.Z, preA.Z, postB.Z, weight)
			);
		}

		public readonly Vector3 CubicInterpolateInTime(Vector3 b, Vector3 preA, Vector3 postB, real_t weight, real_t t, real_t preAT, real_t postBT)
		{
			return new Vector3
			(
				Mathf.CubicInterpolateInTime(X, b.X, preA.X, postB.X, weight, t, preAT, postBT),
				Mathf.CubicInterpolateInTime(Y, b.Y, preA.Y, postB.Y, weight, t, preAT, postBT),
				Mathf.CubicInterpolateInTime(Z, b.Z, preA.Z, postB.Z, weight, t, preAT, postBT)
			);
		}

		public readonly Vector3 BezierInterpolate(Vector3 control1, Vector3 control2, Vector3 end, real_t t)
		{
			return new Vector3
			(
				Mathf.BezierInterpolate(X, control1.X, control2.X, end.X, t),
				Mathf.BezierInterpolate(Y, control1.Y, control2.Y, end.Y, t),
				Mathf.BezierInterpolate(Z, control1.Z, control2.Z, end.Z, t)
			);
		}

		public readonly Vector3 BezierDerivative(Vector3 control1, Vector3 control2, Vector3 end, real_t t)
		{
			return new Vector3(
				Mathf.BezierDerivative(X, control1.X, control2.X, end.X, t),
				Mathf.BezierDerivative(Y, control1.Y, control2.Y, end.Y, t),
				Mathf.BezierDerivative(Z, control1.Z, control2.Z, end.Z, t)
			);
		}

		public readonly Vector3 DirectionTo(Vector3 to)
		{
			return new Vector3(to.X - X, to.Y - Y, to.Z - Z).Normalized();
		}

		public readonly real_t DistanceSquaredTo(Vector3 to)
		{
			return (to - this).LengthSquared();
		}

		public readonly real_t DistanceTo(Vector3 to)
		{
			return (to - this).Length();
		}

		public readonly real_t Dot(Vector3 with)
		{
			return (X * with.X) + (Y * with.Y) + (Z * with.Z);
		}

		public readonly Vector3 Floor()
		{
			return new Vector3(Mathf.Floor(X), Mathf.Floor(Y), Mathf.Floor(Z));
		}

		public readonly Vector3 Inverse()
		{
			return new Vector3(1 / X, 1 / Y, 1 / Z);
		}

		public readonly bool IsFinite()
		{
			return Mathf.IsFinite(X) && Mathf.IsFinite(Y) && Mathf.IsFinite(Z);
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

			return Mathf.Sqrt(x2 + y2 + z2);
		}

		public readonly real_t LengthSquared()
		{
			real_t x2 = X * X;
			real_t y2 = Y * Y;
			real_t z2 = Z * Z;

			return x2 + y2 + z2;
		}

		public readonly Vector3 Lerp(Vector3 to, real_t weight)
		{
			return new Vector3
			(
				Mathf.Lerp(X, to.X, weight),
				Mathf.Lerp(Y, to.Y, weight),
				Mathf.Lerp(Z, to.Z, weight)
			);
		}

		public readonly Vector3 LimitLength(real_t length = 1.0f)
		{
			Vector3 v = this;
			real_t l = Length();

			if (l > 0 && length < l)
			{
				v /= l;
				v *= length;
			}

			return v;
		}

		public readonly Vector3 Max(Vector3 with)
		{
			return new Vector3
			(
				Mathf.Max(X, with.X),
				Mathf.Max(Y, with.Y),
				Mathf.Max(Z, with.Z)
			);
		}

		public readonly Vector3 Max(real_t with)
		{
			return new Vector3
			(
				Mathf.Max(X, with),
				Mathf.Max(Y, with),
				Mathf.Max(Z, with)
			);
		}

		public readonly Vector3 Min(Vector3 with)
		{
			return new Vector3
			(
				Mathf.Min(X, with.X),
				Mathf.Min(Y, with.Y),
				Mathf.Min(Z, with.Z)
			);
		}

		public readonly Vector3 Min(real_t with)
		{
			return new Vector3
			(
				Mathf.Min(X, with),
				Mathf.Min(Y, with),
				Mathf.Min(Z, with)
			);
		}

		public readonly Axis MaxAxisIndex()
		{
			return X < Y ? (Y < Z ? Axis.Z : Axis.Y) : (X < Z ? Axis.Z : Axis.X);
		}

		public readonly Axis MinAxisIndex()
		{
			return X < Y ? (X < Z ? Axis.X : Axis.Z) : (Y < Z ? Axis.Y : Axis.Z);
		}

		public readonly Vector3 MoveToward(Vector3 to, real_t delta)
		{
			Vector3 v = this;
			Vector3 vd = to - v;
			real_t len = vd.Length();
			if (len <= delta || len < Mathf.Epsilon)
				return to;

			return v + (vd / len * delta);
		}

		public readonly Vector3 Normalized()
		{
			Vector3 v = this;
			v.Normalize();
			return v;
		}

		public readonly Basis Outer(Vector3 with)
		{
			return new Basis(
				X * with.X, X * with.Y, X * with.Z,
				Y * with.X, Y * with.Y, Y * with.Z,
				Z * with.X, Z * with.Y, Z * with.Z
			);
		}

		public readonly Vector3 PosMod(real_t mod)
		{
			Vector3 v;
			v.X = Mathf.PosMod(X, mod);
			v.Y = Mathf.PosMod(Y, mod);
			v.Z = Mathf.PosMod(Z, mod);
			return v;
		}

		public readonly Vector3 PosMod(Vector3 modv)
		{
			Vector3 v;
			v.X = Mathf.PosMod(X, modv.X);
			v.Y = Mathf.PosMod(Y, modv.Y);
			v.Z = Mathf.PosMod(Z, modv.Z);
			return v;
		}

		public readonly Vector3 Project(Vector3 onNormal)
		{
			return onNormal * (Dot(onNormal) / onNormal.LengthSquared());
		}

		public readonly Vector3 Reflect(Vector3 normal)
		{
#if DEBUG
			if (!normal.IsNormalized())
			{
				throw new ArgumentException("Argument is not normalized.", nameof(normal));
			}
#endif
			return (2.0f * Dot(normal) * normal) - this;
		}

		public readonly Vector3 Rotated(Vector3 axis, real_t angle)
		{
#if DEBUG
			if (!axis.IsNormalized())
			{
				throw new ArgumentException("Argument is not normalized.", nameof(axis));
			}
#endif
			return new Basis(axis, angle) * this;
		}

		public readonly Vector3 Round()
		{
			return new Vector3(Mathf.Round(X), Mathf.Round(Y), Mathf.Round(Z));
		}

		public readonly Vector3 Sign()
		{
			Vector3 v;
			v.X = Mathf.Sign(X);
			v.Y = Mathf.Sign(Y);
			v.Z = Mathf.Sign(Z);
			return v;
		}

		public readonly real_t SignedAngleTo(Vector3 to, Vector3 axis)
		{
			Vector3 crossTo = Cross(to);
			real_t unsignedAngle = Mathf.Atan2(crossTo.Length(), Dot(to));
			real_t sign = crossTo.Dot(axis);
			return (sign < 0) ? -unsignedAngle : unsignedAngle;
		}

		public readonly Vector3 Slerp(Vector3 to, real_t weight)
		{
			real_t startLengthSquared = LengthSquared();
			real_t endLengthSquared = to.LengthSquared();
			if (startLengthSquared == 0.0 || endLengthSquared == 0.0)
			{
				return Lerp(to, weight);
			}
			Vector3 axis = Cross(to);
			real_t axisLengthSquared = axis.LengthSquared();
			if (axisLengthSquared == 0.0)
			{
				return Lerp(to, weight);
			}
			axis /= Mathf.Sqrt(axisLengthSquared);
			real_t startLength = Mathf.Sqrt(startLengthSquared);
			real_t resultLength = Mathf.Lerp(startLength, Mathf.Sqrt(endLengthSquared), weight);
			real_t angle = AngleTo(to);
			return Rotated(axis, angle * weight) * (resultLength / startLength);
		}

		public readonly Vector3 Slide(Vector3 normal)
		{
			return this - (normal * Dot(normal));
		}

		public readonly Vector3 Snapped(Vector3 step)
		{
			return new Vector3
			(
				Mathf.Snapped(X, step.X),
				Mathf.Snapped(Y, step.Y),
				Mathf.Snapped(Z, step.Z)
			);
		}

		public readonly Vector3 Snapped(real_t step)
		{
			return new Vector3
			(
				Mathf.Snapped(X, step),
				Mathf.Snapped(Y, step),
				Mathf.Snapped(Z, step)
			);
		}

		public readonly Vector2 OctahedronEncode()
		{
			Vector3 n = this;
			n /= Mathf.Abs(n.X) + Mathf.Abs(n.Y) + Mathf.Abs(n.Z);
			Vector2 o;
			if (n.Z >= 0.0f)
			{
				o.X = n.X;
				o.Y = n.Y;
			}
			else
			{
				o.X = (1.0f - Mathf.Abs(n.Y)) * (n.X >= 0.0f ? 1.0f : -1.0f);
				o.Y = (1.0f - Mathf.Abs(n.X)) * (n.Y >= 0.0f ? 1.0f : -1.0f);
			}
			o.X = o.X * 0.5f + 0.5f;
			o.Y = o.Y * 0.5f + 0.5f;
			return o;
		}

		public static Vector3 OctahedronDecode(Vector2 oct)
		{
			var f = new Vector2(oct.X * 2.0f - 1.0f, oct.Y * 2.0f - 1.0f);
			var n = new Vector3(f.X, f.Y, 1.0f - Mathf.Abs(f.X) - Mathf.Abs(f.Y));
			real_t t = Mathf.Clamp(-n.Z, 0.0f, 1.0f);
			n.X += n.X >= 0 ? -t : t;
			n.Y += n.Y >= 0 ? -t : t;
			return n.Normalized();
		}

		private static readonly Vector3 _zero = new Vector3(0, 0, 0);
		private static readonly Vector3 _one = new Vector3(1, 1, 1);
		private static readonly Vector3 _inf = new Vector3(Mathf.Inf, Mathf.Inf, Mathf.Inf);

		private static readonly Vector3 _up = new Vector3(0, 1, 0);
		private static readonly Vector3 _down = new Vector3(0, -1, 0);
		private static readonly Vector3 _right = new Vector3(1, 0, 0);
		private static readonly Vector3 _left = new Vector3(-1, 0, 0);
		private static readonly Vector3 _forward = new Vector3(0, 0, -1);
		private static readonly Vector3 _back = new Vector3(0, 0, 1);

		private static readonly Vector3 _modelLeft = new Vector3(1, 0, 0);
		private static readonly Vector3 _modelRight = new Vector3(-1, 0, 0);
		private static readonly Vector3 _modelTop = new Vector3(0, 1, 0);
		private static readonly Vector3 _modelBottom = new Vector3(0, -1, 0);
		private static readonly Vector3 _modelFront = new Vector3(0, 0, 1);
		private static readonly Vector3 _modelRear = new Vector3(0, 0, -1);

		public static Vector3 Zero { get { return _zero; } }
		public static Vector3 One { get { return _one; } }
		public static Vector3 Inf { get { return _inf; } }

		public static Vector3 Up { get { return _up; } }
		public static Vector3 Down { get { return _down; } }
		public static Vector3 Right { get { return _right; } }
		public static Vector3 Left { get { return _left; } }
		public static Vector3 Forward { get { return _forward; } }
		public static Vector3 Back { get { return _back; } }

		public static Vector3 ModelLeft { get { return _modelLeft; } }
		public static Vector3 ModelRight { get { return _modelRight; } }
		public static Vector3 ModelTop { get { return _modelTop; } }
		public static Vector3 ModelBottom { get { return _modelBottom; } }
		public static Vector3 ModelFront { get { return _modelFront; } }
		public static Vector3 ModelRear { get { return _modelRear; } }

		public Vector3(real_t x, real_t y, real_t z)
		{
			X = x;
			Y = y;
			Z = z;
		}

		public static Vector3 operator +(Vector3 left, Vector3 right)
		{
			left.X += right.X;
			left.Y += right.Y;
			left.Z += right.Z;
			return left;
		}

		public static Vector3 operator -(Vector3 left, Vector3 right)
		{
			left.X -= right.X;
			left.Y -= right.Y;
			left.Z -= right.Z;
			return left;
		}

		public static Vector3 operator -(Vector3 vec)
		{
			vec.X = -vec.X;
			vec.Y = -vec.Y;
			vec.Z = -vec.Z;
			return vec;
		}

		public static Vector3 operator *(Vector3 vec, real_t scale)
		{
			vec.X *= scale;
			vec.Y *= scale;
			vec.Z *= scale;
			return vec;
		}

		public static Vector3 operator *(real_t scale, Vector3 vec)
		{
			vec.X *= scale;
			vec.Y *= scale;
			vec.Z *= scale;
			return vec;
		}

		public static Vector3 operator *(Vector3 left, Vector3 right)
		{
			left.X *= right.X;
			left.Y *= right.Y;
			left.Z *= right.Z;
			return left;
		}

		public static Vector3 operator /(Vector3 vec, real_t divisor)
		{
			vec.X /= divisor;
			vec.Y /= divisor;
			vec.Z /= divisor;
			return vec;
		}

		public static Vector3 operator /(Vector3 vec, Vector3 divisorv)
		{
			vec.X /= divisorv.X;
			vec.Y /= divisorv.Y;
			vec.Z /= divisorv.Z;
			return vec;
		}

		public static Vector3 operator %(Vector3 vec, real_t divisor)
		{
			vec.X %= divisor;
			vec.Y %= divisor;
			vec.Z %= divisor;
			return vec;
		}

		public static Vector3 operator %(Vector3 vec, Vector3 divisorv)
		{
			vec.X %= divisorv.X;
			vec.Y %= divisorv.Y;
			vec.Z %= divisorv.Z;
			return vec;
		}

		public static bool operator ==(Vector3 left, Vector3 right)
		{
			return left.Equals(right);
		}

		public static bool operator !=(Vector3 left, Vector3 right)
		{
			return !left.Equals(right);
		}

		public static bool operator <(Vector3 left, Vector3 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					return left.Z < right.Z;
				}
				return left.Y < right.Y;
			}
			return left.X < right.X;
		}

		public static bool operator >(Vector3 left, Vector3 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					return left.Z > right.Z;
				}
				return left.Y > right.Y;
			}
			return left.X > right.X;
		}

		public static bool operator <=(Vector3 left, Vector3 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					return left.Z <= right.Z;
				}
				return left.Y < right.Y;
			}
			return left.X < right.X;
		}

		public static bool operator >=(Vector3 left, Vector3 right)
		{
			if (left.X == right.X)
			{
				if (left.Y == right.Y)
				{
					return left.Z >= right.Z;
				}
				return left.Y > right.Y;
			}
			return left.X > right.X;
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Vector3 other && Equals(other);
		}

		public readonly bool Equals(Vector3 other)
		{
			return X == other.X && Y == other.Y && Z == other.Z;
		}

		public readonly bool IsEqualApprox(Vector3 other)
		{
			return Mathf.IsEqualApprox(X, other.X) && Mathf.IsEqualApprox(Y, other.Y) && Mathf.IsEqualApprox(Z, other.Z);
		}

		public readonly bool IsZeroApprox()
		{
			return Mathf.IsZeroApprox(X) && Mathf.IsZeroApprox(Y) && Mathf.IsZeroApprox(Z);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(X, Y, Z);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			return $"({X.ToString(format, CultureInfo.InvariantCulture)}, {Y.ToString(format, CultureInfo.InvariantCulture)}, {Z.ToString(format, CultureInfo.InvariantCulture)})";
		}

		internal readonly Vector3 GetAnyPerpendicular()
		{
			if (IsZeroApprox())
			{
				throw new ArgumentException("The Vector3 must not be zero.");
			}
			return Cross((Mathf.Abs(X) <= Mathf.Abs(Y) && Mathf.Abs(X) <= Mathf.Abs(Z)) ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0)).Normalized();
		}
	}
}
