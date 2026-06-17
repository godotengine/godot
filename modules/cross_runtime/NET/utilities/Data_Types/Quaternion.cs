using System;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
	/// <summary>
	/// A unit quaternion used for representing 3D rotations.

	/// </summary>
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct Quaternion : IEquatable<Quaternion>
	{
		public float X;
		public float Y;
		public float Z;
		public float W;

		public float this[int index]
		{
			readonly get
			{
				switch (index)
				{
					case 0: return X;
					case 1: return Y;
					case 2: return Z;
					case 3: return W;
					default: throw new ArgumentOutOfRangeException(nameof(index));
				}
			}
			set
			{
				switch (index)
				{
					case 0: X = value; break;
					case 1: Y = value; break;
					case 2: Z = value; break;
					case 3: W = value; break;
					default: throw new ArgumentOutOfRangeException(nameof(index));
				}
			}
		}



		public readonly float AngleTo(Quaternion to)
		{
			float dot = Dot(to);
			return Acos(Clamp(dot * dot * 2f - 1f, -1f, 1f));
		}

		public readonly Quaternion SphericalCubicInterpolate(Quaternion b, Quaternion preA, Quaternion postB, float weight)
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized");
			if (!b.IsNormalized())
				throw new ArgumentException("Argument is not normalized", nameof(b));
#endif
			Quaternion fromQ = new Basis(this).GetRotationQuaternion();
			Quaternion preQ = new Basis(preA).GetRotationQuaternion();
			Quaternion toQ = new Basis(b).GetRotationQuaternion();
			Quaternion postQ = new Basis(postB).GetRotationQuaternion();

			bool flip1 = Math.Sign(fromQ.Dot(preQ)) < 0;
			preQ = flip1 ? -preQ : preQ;

			bool flip2 = Math.Sign(fromQ.Dot(toQ)) < 0;
			toQ = flip2 ? -toQ : toQ;

			bool flip3 = flip2 ? toQ.Dot(postQ) <= 0 : Math.Sign(toQ.Dot(postQ)) < 0;
			postQ = flip3 ? -postQ : postQ;

			Quaternion lnFrom = new Quaternion(0f, 0f, 0f, 0f);
			Quaternion lnTo = (fromQ.Inverse() * toQ).Log();
			Quaternion lnPre = (fromQ.Inverse() * preQ).Log();
			Quaternion lnPost = (fromQ.Inverse() * postQ).Log();
			Quaternion ln = new Quaternion(
				CubicInterpolate(lnFrom.X, lnTo.X, lnPre.X, lnPost.X, weight),
				CubicInterpolate(lnFrom.Y, lnTo.Y, lnPre.Y, lnPost.Y, weight),
				CubicInterpolate(lnFrom.Z, lnTo.Z, lnPre.Z, lnPost.Z, weight),
				0f
			);
			Quaternion q1 = fromQ * ln.Exp();

			lnFrom = (toQ.Inverse() * fromQ).Log();
			lnTo = new Quaternion(0f, 0f, 0f, 0f);
			lnPre = (toQ.Inverse() * preQ).Log();
			lnPost = (toQ.Inverse() * postQ).Log();
			ln = new Quaternion(
				CubicInterpolate(lnFrom.X, lnTo.X, lnPre.X, lnPost.X, weight),
				CubicInterpolate(lnFrom.Y, lnTo.Y, lnPre.Y, lnPost.Y, weight),
				CubicInterpolate(lnFrom.Z, lnTo.Z, lnPre.Z, lnPost.Z, weight),
				0f
			);
			Quaternion q2 = toQ * ln.Exp();

			return q1.Slerp(q2, weight);
		}

		public readonly Quaternion SphericalCubicInterpolateInTime(Quaternion b, Quaternion preA, Quaternion postB, float weight, float bT, float preAT, float postBT)
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized");
			if (!b.IsNormalized())
				throw new ArgumentException("Argument is not normalized", nameof(b));
#endif
			Quaternion fromQ = new Basis(this).GetRotationQuaternion();
			Quaternion preQ = new Basis(preA).GetRotationQuaternion();
			Quaternion toQ = new Basis(b).GetRotationQuaternion();
			Quaternion postQ = new Basis(postB).GetRotationQuaternion();

			bool flip1 = Math.Sign(fromQ.Dot(preQ)) < 0;
			preQ = flip1 ? -preQ : preQ;

			bool flip2 = Math.Sign(fromQ.Dot(toQ)) < 0;
			toQ = flip2 ? -toQ : toQ;

			bool flip3 = flip2 ? toQ.Dot(postQ) <= 0 : Math.Sign(toQ.Dot(postQ)) < 0;
			postQ = flip3 ? -postQ : postQ;

			Quaternion lnFrom = new Quaternion(0f, 0f, 0f, 0f);
			Quaternion lnTo = (fromQ.Inverse() * toQ).Log();
			Quaternion lnPre = (fromQ.Inverse() * preQ).Log();
			Quaternion lnPost = (fromQ.Inverse() * postQ).Log();
			Quaternion ln = new Quaternion(
				CubicInterpolateInTime(lnFrom.X, lnTo.X, lnPre.X, lnPost.X, weight, bT, preAT, postBT),
				CubicInterpolateInTime(lnFrom.Y, lnTo.Y, lnPre.Y, lnPost.Y, weight, bT, preAT, postBT),
				CubicInterpolateInTime(lnFrom.Z, lnTo.Z, lnPre.Z, lnPost.Z, weight, bT, preAT, postBT),
				0f
			);
			Quaternion q1 = fromQ * ln.Exp();

			lnFrom = (toQ.Inverse() * fromQ).Log();
			lnTo = new Quaternion(0f, 0f, 0f, 0f);
			lnPre = (toQ.Inverse() * preQ).Log();
			lnPost = (toQ.Inverse() * postQ).Log();
			ln = new Quaternion(
				CubicInterpolateInTime(lnFrom.X, lnTo.X, lnPre.X, lnPost.X, weight, bT, preAT, postBT),
				CubicInterpolateInTime(lnFrom.Y, lnTo.Y, lnPre.Y, lnPost.Y, weight, bT, preAT, postBT),
				CubicInterpolateInTime(lnFrom.Z, lnTo.Z, lnPre.Z, lnPost.Z, weight, bT, preAT, postBT),
				0f
			);
			Quaternion q2 = toQ * ln.Exp();

			return q1.Slerp(q2, weight);
		}

		public readonly float Dot(Quaternion b)
		{
			return (X * b.X) + (Y * b.Y) + (Z * b.Z) + (W * b.W);
		}

		public readonly Quaternion Exp()
		{
			Vector3 v = new Vector3(X, Y, Z);
			float theta = Length(v);
			v = Normalize(v);

			if (theta < Epsilon || !IsNormalizedVector(v))
				return new Quaternion(0f, 0f, 0f, 1f);

			return new Quaternion(v, theta);
		}

		public readonly float GetAngle()
		{
			return 2f * Acos(W);
		}

		public readonly Vector3 GetAxis()
		{
			if (Abs(W) > 1f - Epsilon)
				return new Vector3(X, Y, Z);

			float r = 1f / Sqrt(1f - W * W);
			return new Vector3(X * r, Y * r, Z * r);
		}

		public readonly Vector3 GetEuler(EulerOrder order = EulerOrder.EULER_ORDER_YXZ)
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized.");
#endif
			return new Basis(this).GetEuler(order);
		}

		public readonly Quaternion Inverse()
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized.");
#endif
			return new Quaternion(-X, -Y, -Z, W);
		}

		public readonly bool IsFinite()
		{
			return IsFinite(X) && IsFinite(Y) && IsFinite(Z) && IsFinite(W);
		}

		public readonly bool IsNormalized()
		{
			return IsEqualApprox(LengthSquared(), 1f, Epsilon);
		}

		public readonly Quaternion Log()
		{
			Vector3 v = GetAxis();
			float angle = GetAngle();
			v = Mul(v, angle);
			return new Quaternion(v.X, v.Y, v.Z, 0f);
		}

		public readonly float Length()
		{
			return Sqrt(LengthSquared());
		}

		public readonly float LengthSquared()
		{
			return Dot(this);
		}

		public readonly Quaternion Normalized()
		{
			float len = Length();
			if (len == 0f)
				return new Quaternion(0f, 0f, 0f, 0f);

			return this / len;
		}

		public readonly Quaternion Slerp(Quaternion to, float weight)
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized.");
			if (!to.IsNormalized())
				throw new ArgumentException("Argument is not normalized.", nameof(to));
#endif
			float cosom = Dot(to);
			Quaternion to1;

			if (cosom < 0.0f)
			{
				cosom = -cosom;
				to1 = -to;
			}
			else
			{
				to1 = to;
			}

			float scale0, scale1;
			if (1.0f - cosom > Epsilon)
			{
				float omega = Acos(cosom);
				float sinom = Sin(omega);
				scale0 = Sin((1.0f - weight) * omega) / sinom;
				scale1 = Sin(weight * omega) / sinom;
			}
			else
			{
				scale0 = 1.0f - weight;
				scale1 = weight;
			}

			return new Quaternion(
				(scale0 * X) + (scale1 * to1.X),
				(scale0 * Y) + (scale1 * to1.Y),
				(scale0 * Z) + (scale1 * to1.Z),
				(scale0 * W) + (scale1 * to1.W)
			);
		}

		public readonly Quaternion Slerpni(Quaternion to, float weight)
		{
#if DEBUG
			if (!IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized");
			if (!to.IsNormalized())
				throw new ArgumentException("Argument is not normalized", nameof(to));
#endif
			float dot = Dot(to);

			if (Abs(dot) > 0.9999f)
				return this;

			float theta = Acos(dot);
			float sinT = 1.0f / Sin(theta);
			float newFactor = Sin(weight * theta) * sinT;
			float invFactor = Sin((1.0f - weight) * theta) * sinT;

			return new Quaternion(
				(invFactor * X) + (newFactor * to.X),
				(invFactor * Y) + (newFactor * to.Y),
				(invFactor * Z) + (newFactor * to.Z),
				(invFactor * W) + (newFactor * to.W)
			);
		}

		private static readonly Quaternion _identity = new Quaternion(0f, 0f, 0f, 1f);
		public static Quaternion Identity => _identity;

		public Quaternion(float x, float y, float z, float w)
		{
			X = x;
			Y = y;
			Z = z;
			W = w;
		}

		public Quaternion(Basis basis)
		{
			this = basis.GetQuaternion();
		}

		public Quaternion(Vector3 axis, float angle)
		{
			float d = Length(axis);

			if (d == 0f)
			{
				X = 0f;
				Y = 0f;
				Z = 0f;
				W = 0f;
				return;
			}

			float sin = Sin(angle * 0.5f);
			float cos = Cos(angle * 0.5f);
			float s = sin / d;

			X = axis.X * s;
			Y = axis.Y * s;
			Z = axis.Z * s;
			W = cos;
		}

		public Quaternion(Vector3 arcFrom, Vector3 arcTo)
		{
			const float AlmostOne = 0.99999975f;

			Vector3 n0 = Normalize(arcFrom);
			Vector3 n1 = Normalize(arcTo);
			float d = Dot(n0, n1);

			if (Abs(d) > AlmostOne)
			{
				if (d >= 0.0f)
				{
					X = 0f;
					Y = 0f;
					Z = 0f;
					W = 1f;
					return;
				}

				Vector3 axis = GetAnyPerpendicular(n0);
				X = axis.X;
				Y = axis.Y;
				Z = axis.Z;
				W = 0f;
			}
			else
			{
				Vector3 c = Cross(n0, n1);
				float s = Sqrt((1.0f + d) * 2.0f);
				float rs = 1.0f / s;

				X = c.X * rs;
				Y = c.Y * rs;
				Z = c.Z * rs;
				W = s * 0.5f;
			}

			this = Normalized();
		}

		public static Quaternion FromEuler(Vector3 eulerYXZ)
		{
			float halfA1 = eulerYXZ.Y * 0.5f;
			float halfA2 = eulerYXZ.X * 0.5f;
			float halfA3 = eulerYXZ.Z * 0.5f;

			float sinA1 = Sin(halfA1);
			float cosA1 = Cos(halfA1);
			float sinA2 = Sin(halfA2);
			float cosA2 = Cos(halfA2);
			float sinA3 = Sin(halfA3);
			float cosA3 = Cos(halfA3);

			return new Quaternion(
				(sinA1 * cosA2 * sinA3) + (cosA1 * sinA2 * cosA3),
				(sinA1 * cosA2 * cosA3) - (cosA1 * sinA2 * sinA3),
				(cosA1 * cosA2 * sinA3) - (sinA1 * sinA2 * cosA3),
				(sinA1 * sinA2 * sinA3) + (cosA1 * cosA2 * cosA3)
			);
		}

		public static Quaternion operator *(Quaternion left, Quaternion right)
		{
			return new Quaternion(
				(left.W * right.X) + (left.X * right.W) + (left.Y * right.Z) - (left.Z * right.Y),
				(left.W * right.Y) + (left.Y * right.W) + (left.Z * right.X) - (left.X * right.Z),
				(left.W * right.Z) + (left.Z * right.W) + (left.X * right.Y) - (left.Y * right.X),
				(left.W * right.W) - (left.X * right.X) - (left.Y * right.Y) - (left.Z * right.Z)
			);
		}

		public static Vector3 operator *(Quaternion quaternion, Vector3 vector)
		{
#if DEBUG
			if (!quaternion.IsNormalized())
				throw new InvalidOperationException("Quaternion is not normalized.");
#endif
			Vector3 u = new Vector3(quaternion.X, quaternion.Y, quaternion.Z);
			Vector3 uv = Cross(u, vector);
			Vector3 term = Add(Mul(uv, quaternion.W), Cross(u, uv));
			return Add(vector, Mul(term, 2f));
		}

		public static Vector3 operator *(Vector3 vector, Quaternion quaternion)
		{
			return quaternion.Inverse() * vector;
		}

		public static Quaternion operator +(Quaternion left, Quaternion right)
		{
			return new Quaternion(left.X + right.X, left.Y + right.Y, left.Z + right.Z, left.W + right.W);
		}

		public static Quaternion operator -(Quaternion left, Quaternion right)
		{
			return new Quaternion(left.X - right.X, left.Y - right.Y, left.Z - right.Z, left.W - right.W);
		}

		public static Quaternion operator -(Quaternion quat)
		{
			return new Quaternion(-quat.X, -quat.Y, -quat.Z, -quat.W);
		}

		public static Quaternion operator *(Quaternion left, float right)
		{
			return new Quaternion(left.X * right, left.Y * right, left.Z * right, left.W * right);
		}

		public static Quaternion operator *(float left, Quaternion right)
		{
			return new Quaternion(right.X * left, right.Y * left, right.Z * left, right.W * left);
		}

		public static Quaternion operator /(Quaternion left, float right)
		{
			return left * (1.0f / right);
		}

		public static bool operator ==(Quaternion left, Quaternion right) => left.Equals(right);
		public static bool operator !=(Quaternion left, Quaternion right) => !left.Equals(right);

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Quaternion other && Equals(other);
		}

		public readonly bool Equals(Quaternion other)
		{
			return X == other.X && Y == other.Y && Z == other.Z && W == other.W;
		}

		public readonly bool IsEqualApprox(Quaternion other)
		{
			return IsEqualApprox(X, other.X) &&
				   IsEqualApprox(Y, other.Y) &&
				   IsEqualApprox(Z, other.Z) &&
				   IsEqualApprox(W, other.W);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(X, Y, Z, W);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			string f = string.IsNullOrEmpty(format) ? "G" : format!;
			return $"({X.ToString(f, CultureInfo.InvariantCulture)}, {Y.ToString(f, CultureInfo.InvariantCulture)}, {Z.ToString(f, CultureInfo.InvariantCulture)}, {W.ToString(f, CultureInfo.InvariantCulture)})";
		}

		private static float Dot(Quaternion a, Quaternion b)
		{
			return (a.X * b.X) + (a.Y * b.Y) + (a.Z * b.Z) + (a.W * b.W);
		}

		private static float Dot(Vector3 a, Vector3 b)
		{
			return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
		}

		private static float LengthSquared(Vector3 v)
		{
			return Dot(v, v);
		}

		private static float Length(Vector3 v)
		{
			return Sqrt(LengthSquared(v));
		}

		private static Vector3 Normalize(Vector3 v)
		{
			float len = Length(v);
			if (len == 0f)
				return new Vector3(0f, 0f, 0f);

			return new Vector3(v.X / len, v.Y / len, v.Z / len);
		}

		private static bool IsNormalizedVector(Vector3 v)
		{
			return IsEqualApprox(LengthSquared(v), 1f, Epsilon);
		}

		private static Vector3 GetAnyPerpendicular(Vector3 v)
		{
			Vector3 basis = Abs(v.X) < Abs(v.Y)
				? (Abs(v.X) < Abs(v.Z) ? new Vector3(1f, 0f, 0f) : new Vector3(0f, 0f, 1f))
				: (Abs(v.Y) < Abs(v.Z) ? new Vector3(0f, 1f, 0f) : new Vector3(0f, 0f, 1f));

			return Normalize(Cross(v, basis));
		}

		private static Vector3 Cross(Vector3 a, Vector3 b)
		{
			return new Vector3(
				a.Y * b.Z - a.Z * b.Y,
				a.Z * b.X - a.X * b.Z,
				a.X * b.Y - a.Y * b.X
			);
		}

		private static Vector3 Add(Vector3 a, Vector3 b)
		{
			return new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
		}

		private static Vector3 Mul(Vector3 v, float s)
		{
			return new Vector3(v.X * s, v.Y * s, v.Z * s);
		}

		private static float CubicInterpolate(float preA, float a, float b, float postB, float weight)
		{
			float t = weight;
			float t2 = t * t;
			float t3 = t2 * t;
			return 0.5f * (
				(2f * a) +
				(-preA + b) * t +
				(2f * preA - 5f * a + 4f * b - postB) * t2 +
				(-preA + 3f * a - 3f * b + postB) * t3
			);
		}

		private static float CubicInterpolateInTime(float preA, float a, float b, float postB, float weight, float bT, float preAT, float postBT)
		{
			float t = weight;
			float t2 = t * t;
			float t3 = t2 * t;

			float m0 = (b - preA) / (bT - preAT);
			float m1 = (postB - a) / (postBT - bT);

			float t0 = t3 - 2f * t2 + t;
			float t1 = -2f * t3 + 3f * t2;
			float t2c = t3 - t2;

			return a * (2f * t3 - 3f * t2 + 1f)
				 + m0 * t0 * (bT - a)
				 + b * (-2f * t3 + 3f * t2)
				 + m1 * t2c * (postBT - bT);
		}

		private static float Sin(float v) => MathF.Sin(v);
		private static float Cos(float v) => MathF.Cos(v);
		private static float Tan(float v) => MathF.Tan(v);
		private static float Asin(float v) => MathF.Asin(v);
		private static float Acos(float v) => MathF.Acos(v);
		private static float Sqrt(float v) => MathF.Sqrt(v);
		private static float Abs(float v) => MathF.Abs(v);
		private static float Clamp(float v, float min, float max) => Math.Clamp(v, min, max);

		private static bool IsFinite(float value)
		{
			return !float.IsNaN(value) && !float.IsInfinity(value);
		}

		private static bool IsZeroApprox(float value)
		{
			return Abs(value) <= Epsilon;
		}


		public static bool IsEqualApprox(real_t a, real_t b)
		{
			return IsEqualApprox(a, b, Mathf.Epsilon);
		}

		public static bool IsEqualApprox(real_t a, real_t b, real_t tolerance)
		{
			return Math.Abs(a - b) <= tolerance;
		}

		private const float Epsilon = 0.00001f;
	}
}
