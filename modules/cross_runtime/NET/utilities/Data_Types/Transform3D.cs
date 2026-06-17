using System;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

#nullable enable
namespace Godot
{
	[Serializable]
	[StructLayout(LayoutKind.Sequential)]
	public struct Transform3D : IEquatable<Transform3D>
	{
		public Basis Basis;
		public Vector3 Origin;

		public Vector3 this[int column]
		{
			readonly get
			{
				switch (column)
				{
					case 0:
						return Basis.Column0;
					case 1:
						return Basis.Column1;
					case 2:
						return Basis.Column2;
					case 3:
						return Origin;
					default:
						throw new ArgumentOutOfRangeException(nameof(column));
				}
			}
			set
			{
				switch (column)
				{
					case 0:
						Basis.Column0 = value;
						return;
					case 1:
						Basis.Column1 = value;
						return;
					case 2:
						Basis.Column2 = value;
						return;
					case 3:
						Origin = value;
						return;
					default:
						throw new ArgumentOutOfRangeException(nameof(column));
				}
			}
		}

		public real_t this[int column, int row]
		{
			readonly get
			{
				if (column == 3)
				{
					return Origin[row];
				}
				return Basis[column, row];
			}
			set
			{
				if (column == 3)
				{
					Origin[row] = value;
					return;
				}
				Basis[column, row] = value;
			}
		}

		public readonly Transform3D AffineInverse()
		{
			Basis basisInv = Basis.Inverse();
			return new Transform3D(basisInv, basisInv * -Origin);
		}

		public readonly Transform3D InterpolateWith(Transform3D transform, real_t weight)
		{
			Vector3 sourceScale = Basis.Scale;
			Quaternion sourceRotation = Basis.GetRotationQuaternion();
			Vector3 sourceLocation = Origin;

			Vector3 destinationScale = transform.Basis.Scale;
			Quaternion destinationRotation = transform.Basis.GetRotationQuaternion();
			Vector3 destinationLocation = transform.Origin;

			var interpolated = new Transform3D();
			Quaternion quaternion = sourceRotation.Slerp(destinationRotation, weight).Normalized();
			Vector3 scale = sourceScale.Lerp(destinationScale, weight);
			interpolated.Basis.SetQuaternionScale(quaternion, scale);
			interpolated.Origin = sourceLocation.Lerp(destinationLocation, weight);

			return interpolated;
		}

		public readonly Transform3D Inverse()
		{
			Basis basisTr = Basis.Transposed();
			return new Transform3D(basisTr, basisTr * -Origin);
		}

		public readonly bool IsFinite()
		{
			return Basis.IsFinite() && Origin.IsFinite();
		}

		public readonly Transform3D LookingAt(Vector3 target, Vector3? up = null, bool useModelFront = false)
		{
			Transform3D t = this;
			t.SetLookAt(Origin, target, up ?? Vector3.Up, useModelFront);
			return t;
		}

		[EditorBrowsable(EditorBrowsableState.Never)]
		public readonly Transform3D LookingAt(Vector3 target, Vector3 up)
		{
			return LookingAt(target, up, false);
		}

		public readonly Transform3D Orthonormalized()
		{
			return new Transform3D(Basis.Orthonormalized(), Origin);
		}

		public readonly Transform3D Rotated(Vector3 axis, real_t angle)
		{
			return new Transform3D(new Basis(axis, angle), new Vector3()) * this;
		}

		public readonly Transform3D RotatedLocal(Vector3 axis, real_t angle)
		{
			Basis tmpBasis = new Basis(axis, angle);
			return new Transform3D(Basis * tmpBasis, Origin);
		}

		public readonly Transform3D Scaled(Vector3 scale)
		{
			return new Transform3D(Basis.Scaled(scale), Origin * scale);
		}

		public readonly Transform3D ScaledLocal(Vector3 scale)
		{
			Basis tmpBasis = Basis.FromScale(scale);
			return new Transform3D(Basis * tmpBasis, Origin);
		}

		private void SetLookAt(Vector3 eye, Vector3 target, Vector3 up, bool useModelFront = false)
		{
			Basis = Basis.LookingAt(target - eye, up, useModelFront);
			Origin = eye;
		}

		public readonly Transform3D Translated(Vector3 offset)
		{
			return new Transform3D(Basis, Origin + offset);
		}

		public readonly Transform3D TranslatedLocal(Vector3 offset)
		{
			return new Transform3D(Basis, new Vector3
			(
				Origin[0] + Basis.Row0.Dot(offset),
				Origin[1] + Basis.Row1.Dot(offset),
				Origin[2] + Basis.Row2.Dot(offset)
			));
		}

		private static readonly Transform3D _identity = new Transform3D(Basis.Identity, Vector3.Zero);
		private static readonly Transform3D _flipX = new Transform3D(new Basis(-1, 0, 0, 0, 1, 0, 0, 0, 1), Vector3.Zero);
		private static readonly Transform3D _flipY = new Transform3D(new Basis(1, 0, 0, 0, -1, 0, 0, 0, 1), Vector3.Zero);
		private static readonly Transform3D _flipZ = new Transform3D(new Basis(1, 0, 0, 0, 1, 0, 0, 0, -1), Vector3.Zero);

		public static Transform3D Identity { get { return _identity; } }
		public static Transform3D FlipX { get { return _flipX; } }
		public static Transform3D FlipY { get { return _flipY; } }
		public static Transform3D FlipZ { get { return _flipZ; } }

		public Transform3D(Vector3 column0, Vector3 column1, Vector3 column2, Vector3 origin)
		{
			Basis = new Basis(column0, column1, column2);
			Origin = origin;
		}

		public Transform3D(real_t xx, real_t yx, real_t zx, real_t xy, real_t yy, real_t zy, real_t xz, real_t yz, real_t zz, real_t ox, real_t oy, real_t oz)
		{
			Basis = new Basis(xx, yx, zx, xy, yy, zy, xz, yz, zz);
			Origin = new Vector3(ox, oy, oz);
		}

		public Transform3D(Basis basis, Vector3 origin)
		{
			Basis = basis;
			Origin = origin;
		}

		public Transform3D(Projection projection)
		{
			Basis = new Basis
			(
				projection.X.X, projection.Y.X, projection.Z.X,
				projection.X.Y, projection.Y.Y, projection.Z.Y,
				projection.X.Z, projection.Y.Z, projection.Z.Z
			);
			Origin = new Vector3
			(
				projection.W.X,
				projection.W.Y,
				projection.W.Z
			);
		}

		public static Transform3D operator *(Transform3D left, Transform3D right)
		{
			left.Origin = left * right.Origin;
			left.Basis *= right.Basis;
			return left;
		}

		public static Vector3 operator *(Transform3D transform, Vector3 vector)
		{
			return new Vector3
			(
				transform.Basis.Row0.Dot(vector) + transform.Origin.X,
				transform.Basis.Row1.Dot(vector) + transform.Origin.Y,
				transform.Basis.Row2.Dot(vector) + transform.Origin.Z
			);
		}

		public static Vector3 operator *(Vector3 vector, Transform3D transform)
		{
			Vector3 vInv = vector - transform.Origin;

			return new Vector3
			(
				(transform.Basis.Row0[0] * vInv.X) + (transform.Basis.Row1[0] * vInv.Y) + (transform.Basis.Row2[0] * vInv.Z),
				(transform.Basis.Row0[1] * vInv.X) + (transform.Basis.Row1[1] * vInv.Y) + (transform.Basis.Row2[1] * vInv.Z),
				(transform.Basis.Row0[2] * vInv.X) + (transform.Basis.Row1[2] * vInv.Y) + (transform.Basis.Row2[2] * vInv.Z)
			);
		}

		public static AABB operator *(Transform3D transform, AABB aabb)
		{
			Vector3 min = aabb.Position;
			Vector3 max = aabb.Position + aabb.Size;

			Vector3 tmin = transform.Origin;
			Vector3 tmax = transform.Origin;
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					real_t e = transform.Basis[j][i] * min[j];
					real_t f = transform.Basis[j][i] * max[j];
					if (e < f)
					{
						tmin[i] += e;
						tmax[i] += f;
					}
					else
					{
						tmin[i] += f;
						tmax[i] += e;
					}
				}
			}

			return new AABB(tmin, tmax - tmin);
		}

		public static AABB operator *(AABB aabb, Transform3D transform)
		{
			Vector3 pos = new Vector3(aabb.Position.X + aabb.Size.X, aabb.Position.Y + aabb.Size.Y, aabb.Position.Z + aabb.Size.Z) * transform;
			Vector3 to1 = new Vector3(aabb.Position.X + aabb.Size.X, aabb.Position.Y + aabb.Size.Y, aabb.Position.Z) * transform;
			Vector3 to2 = new Vector3(aabb.Position.X + aabb.Size.X, aabb.Position.Y, aabb.Position.Z + aabb.Size.Z) * transform;
			Vector3 to3 = new Vector3(aabb.Position.X + aabb.Size.X, aabb.Position.Y, aabb.Position.Z) * transform;
			Vector3 to4 = new Vector3(aabb.Position.X, aabb.Position.Y + aabb.Size.Y, aabb.Position.Z + aabb.Size.Z) * transform;
			Vector3 to5 = new Vector3(aabb.Position.X, aabb.Position.Y + aabb.Size.Y, aabb.Position.Z) * transform;
			Vector3 to6 = new Vector3(aabb.Position.X, aabb.Position.Y, aabb.Position.Z + aabb.Size.Z) * transform;
			Vector3 to7 = new Vector3(aabb.Position.X, aabb.Position.Y, aabb.Position.Z) * transform;

			return new AABB(pos, new Vector3()).Expand(to1).Expand(to2).Expand(to3).Expand(to4).Expand(to5).Expand(to6).Expand(to7);
		}

		public static Plane operator *(Transform3D transform, Plane plane)
		{
			Basis bInvTrans = transform.Basis.Inverse().Transposed();

			Vector3 point = transform * (plane.Normal * plane.D);
			Vector3 normal = (bInvTrans * plane.Normal).Normalized();

			real_t d = normal.Dot(point);
			return new Plane(normal, d);
		}

		public static Plane operator *(Plane plane, Transform3D transform)
		{
			Transform3D tInv = transform.AffineInverse();
			Basis bTrans = transform.Basis.Transposed();

			Vector3 point = tInv * (plane.Normal * plane.D);
			Vector3 normal = (bTrans * plane.Normal).Normalized();

			real_t d = normal.Dot(point);
			return new Plane(normal, d);
		}

		public static Vector3[] operator *(Transform3D transform, Vector3[] array)
		{
			Vector3[] newArray = new Vector3[array.Length];

			for (int i = 0; i < array.Length; i++)
			{
				newArray[i] = transform * array[i];
			}

			return newArray;
		}

		public static Vector3[] operator *(Vector3[] array, Transform3D transform)
		{
			Vector3[] newArray = new Vector3[array.Length];

			for (int i = 0; i < array.Length; i++)
			{
				newArray[i] = array[i] * transform;
			}

			return newArray;
		}

		public static bool operator ==(Transform3D left, Transform3D right)
		{
			return left.Equals(right);
		}

		public static bool operator !=(Transform3D left, Transform3D right)
		{
			return !left.Equals(right);
		}

		public override readonly bool Equals([NotNullWhen(true)] object? obj)
		{
			return obj is Transform3D other && Equals(other);
		}

		public readonly bool Equals(Transform3D other)
		{
			return Basis.Equals(other.Basis) && Origin.Equals(other.Origin);
		}

		public readonly bool IsEqualApprox(Transform3D other)
		{
			return Basis.IsEqualApprox(other.Basis) && Origin.IsEqualApprox(other.Origin);
		}

		public override readonly int GetHashCode()
		{
			return HashCode.Combine(Basis, Origin);
		}

		public override readonly string ToString() => ToString(null);

		public readonly string ToString(string? format)
		{
			return $"[X: {Basis.X.ToString(format)}, Y: {Basis.Y.ToString(format)}, Z: {Basis.Z.ToString(format)}, O: {Origin.ToString(format)}]";
		}
	}
}
