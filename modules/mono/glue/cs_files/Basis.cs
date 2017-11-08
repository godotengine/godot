using System;
using System.Runtime.InteropServices;

namespace Godot
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Basis : IEquatable<Basis>
    {
        private static readonly Basis identity = new Basis
        (
            new Vector3(1f, 0f, 0f),
            new Vector3(0f, 1f, 0f),
            new Vector3(0f, 0f, 1f)
        );

        private static readonly Basis[] orthoBases = new Basis[24]
        {
            new Basis(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
            new Basis(0f, -1f, 0f, 1f, 0f, 0f, 0f, 0f, 1f),
            new Basis(-1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, 1f),
            new Basis(0f, 1f, 0f, -1f, 0f, 0f, 0f, 0f, 1f),
            new Basis(1f, 0f, 0f, 0f, 0f, -1f, 0f, 1f, 0f),
            new Basis(0f, 0f, 1f, 1f, 0f, 0f, 0f, 1f, 0f),
            new Basis(-1f, 0f, 0f, 0f, 0f, 1f, 0f, 1f, 0f),
            new Basis(0f, 0f, -1f, -1f, 0f, 0f, 0f, 1f, 0f),
            new Basis(1f, 0f, 0f, 0f, -1f, 0f, 0f, 0f, -1f),
            new Basis(0f, 1f, 0f, 1f, 0f, 0f, 0f, 0f, -1f),
            new Basis(-1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, -1f),
            new Basis(0f, -1f, 0f, -1f, 0f, 0f, 0f, 0f, -1f),
            new Basis(1f, 0f, 0f, 0f, 0f, 1f, 0f, -1f, 0f),
            new Basis(0f, 0f, -1f, 1f, 0f, 0f, 0f, -1f, 0f),
            new Basis(-1f, 0f, 0f, 0f, 0f, -1f, 0f, -1f, 0f),
            new Basis(0f, 0f, 1f, -1f, 0f, 0f, 0f, -1f, 0f),
            new Basis(0f, 0f, 1f, 0f, 1f, 0f, -1f, 0f, 0f),
            new Basis(0f, -1f, 0f, 0f, 0f, 1f, -1f, 0f, 0f),
            new Basis(0f, 0f, -1f, 0f, -1f, 0f, -1f, 0f, 0f),
            new Basis(0f, 1f, 0f, 0f, 0f, -1f, -1f, 0f, 0f),
            new Basis(0f, 0f, 1f, 0f, -1f, 0f, 1f, 0f, 0f),
            new Basis(0f, 1f, 0f, 0f, 0f, 1f, 1f, 0f, 0f),
            new Basis(0f, 0f, -1f, 0f, 1f, 0f, 1f, 0f, 0f),
            new Basis(0f, -1f, 0f, 0f, 0f, -1f, 1f, 0f, 0f)
        };

        public Vector3 x;
        public Vector3 y;
        public Vector3 z;

        public static Basis Identity
        {
            get { return identity; }
        }

        public Vector3 Scale
        {
            get
            {
                return new Vector3
                (
                    new Vector3(this[0, 0], this[1, 0], this[2, 0]).length(),
                    new Vector3(this[0, 1], this[1, 1], this[2, 1]).length(),
                    new Vector3(this[0, 2], this[1, 2], this[2, 2]).length()
                );
            }
        }

        public Vector3 this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return x;
                    case 1:
                        return y;
                    case 2:
                        return z;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        x = value;
                        return;
                    case 1:
                        y = value;
                        return;
                    case 2:
                        z = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public float this[int index, int axis]
        {
            get
            {
                switch (index)
                {
                    case 0:
                        return x[axis];
                    case 1:
                        return y[axis];
                    case 2:
                        return z[axis];
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
            set
            {
                switch (index)
                {
                    case 0:
                        x[axis] = value;
                        return;
                    case 1:
                        y[axis] = value;
                        return;
                    case 2:
                        z[axis] = value;
                        return;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

		/// <summary>
		/// Create a matrix from 3 axis vectors
		/// </summary>
		/// <param name="xAxis">X-axis vector</param>
		/// <param name="yAxis">Y-axis vector</param>
		/// <param name="zAxis">Z-axis vector</param>
		/// <returns>Matrix from vectors</returns>
        internal static Basis CreateFromAxis(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            return new Basis
            (
                new Vector3(xAxis.x, yAxis.x, zAxis.x),
                new Vector3(xAxis.y, yAxis.y, zAxis.y),
                new Vector3(xAxis.z, yAxis.z, zAxis.z)
            );
        }

		/// <summary>
		/// Calculate the determinant of the matrix
		/// </summary>
		/// <returns>Determinant of the matrix</returns>
        public float Determinant()
        {
            return this[0, 0] * (this[1, 1] * this[2, 2] - this[2, 1] * this[1, 2]) -
                    this[1, 0] * (this[0, 1] * this[2, 2] - this[2, 1] * this[0, 2]) +
                    this[2, 0] * (this[0, 1] * this[1, 2] - this[1, 1] * this[0, 2]);
        }

		// TODO
		/// <remarks>Undocumented</remarks>
		public Vector3 GetAxis(int axis)
        {
            return new Vector3(this[0, axis], this[1, axis], this[2, axis]);
        }

		/// <summary>
		/// Assuming that the matrix is a proper rotation matrix (orthonormal matrix with determinant +1), return Euler angles
		/// </summary>
		/// <returns>Rotation angle in the (x,y,z) format</returns>
        public Vector3 GetEuler()
        {
            Basis m = this.Orthonormalized();

            Vector3 euler;
            euler.z = 0.0f;

            float mxy = m.y[2];


            if (mxy < 1.0f)
            {
                if (mxy > -1.0f)
                {
                    euler.x = Mathf.asin(-mxy);
                    euler.y = Mathf.atan2(m.x[2], m.z[2]);
                    euler.z = Mathf.atan2(m.y[0], m.y[1]);
                }
                else
                {
                    euler.x = Mathf.PI * 0.5f;
                    euler.y = -Mathf.atan2(-m.x[1], m.x[0]);
                }
            }
            else
            {
                euler.x = -Mathf.PI * 0.5f;
                euler.y = -Mathf.atan2(m.x[1], m.x[0]);
            }

            return euler;
        }

		/// <summary>
		/// This function considers a discretization of rotations into 24 points on unit sphere, lying along the vectors (x,y,z) with each component being either -1,0 or 1, and returns the index of the point best representing the orientation of the object
		/// </summary>
		/// <remarks>Mainly used by grid map editor</remarks>
		/// <returns>Index of the point best representing the orientation of the object</returns>
		public int GetOrthagonalIndex()
        {
            Basis orth = this;

            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    float v = orth[i, j];

                    if (v > 0.5f)
                        v = 1.0f;
                    else if (v < -0.5f)
                        v = -1.0f;
                    else
                        v = 0f;

                    orth[i, j] = v;
                }
            }

            for (int i = 0; i < 24; i++)
            {
                if (orthoBases[i] == orth)
                    return i;
            }

            return 0;
        }

		/// <summary>
		/// Inverts a matrix
		/// </summary>
		/// <returns>Inverse of the matrix</returns>
        public Basis Inverse()
        {
            Basis inv = this;

            float[] co = new float[3]
            {
                inv[1, 1] * inv[2, 2] - inv[1, 2] * inv[2, 1],
                inv[1, 2] * inv[2, 0] - inv[1, 0] * inv[2, 2],
                inv[1, 0] * inv[2, 1] - inv[1, 1] * inv[2, 0]
            };

            float det = inv[0, 0] * co[0] + inv[0, 1] * co[1] + inv[0, 2] * co[2];

            if (det == 0)
            {
                return new Basis
                (
                    float.NaN, float.NaN, float.NaN,
                    float.NaN, float.NaN, float.NaN,
                    float.NaN, float.NaN, float.NaN
                );
            }

            float s = 1.0f / det;

            inv = new Basis
            (
                co[0] * s,
                inv[0, 2] * inv[2, 1] - inv[0, 1] * inv[2, 2] * s,
                inv[0, 1] * inv[1, 2] - inv[0, 2] * inv[1, 1] * s,
                co[1] * s,
                inv[0, 0] * inv[2, 2] - inv[0, 2] * inv[2, 0] * s,
                inv[0, 2] * inv[1, 0] - inv[0, 0] * inv[1, 2] * s,
                co[2] * s,
                inv[0, 1] * inv[2, 0] - inv[0, 0] * inv[2, 1] * s,
                inv[0, 0] * inv[1, 1] - inv[0, 1] * inv[1, 0] * s
            );

            return inv;
        }

		/// <summary>
		/// Performs a Gram-Schmidt orthonormalization on the basis of the matrix
		/// </summary>
		/// <remarks>Useful to call from time to time to avoid rounding error for orthogonal matrices</remarks>
		/// <returns>orthonormalized version of the matrix</returns>
		public Basis Orthonormalized()
        {
            Vector3 xAxis = GetAxis(0);
            Vector3 yAxis = GetAxis(1);
            Vector3 zAxis = GetAxis(2);

            xAxis.normalize();
            yAxis = (yAxis - xAxis * (xAxis.dot(yAxis)));
            yAxis.normalize();
            zAxis = (zAxis - xAxis * (xAxis.dot(zAxis)) - yAxis * (yAxis.dot(zAxis)));
            zAxis.normalize();

            return Basis.CreateFromAxis(xAxis, yAxis, zAxis);
		}

		/// <summary>
		/// Introduce an additional rotation around the given axis by phi (radians) 
		/// </summary>
		/// <param name="axis">Axis of rotation</param>
		/// <param name="phi">Amount of rotation (radians)</param>
		/// <remarks>Only relevant when the matrix is being used as a part of Transform.</remarks>
		/// <returns>Rotated vector</returns>
		public Basis Rotated(Vector3 axis, float phi)
        {
            return new Basis(axis, phi) * this;
        }

		/// <summary>
		/// Scales a vector by a 3D scaling factor
		/// </summary>
		/// <param name="scale">3D scaling factor</param>
		/// <remarks>Only relevant when the matrix is being used as a part of Transform.</remarks>
		/// <returns>Scaled vector</returns>
		public Basis Scaled(Vector3 scale)
        {
            Basis m = this;

            m[0, 0] *= scale.x;
            m[0, 1] *= scale.x;
            m[0, 2] *= scale.x;
            m[1, 0] *= scale.y;
            m[1, 1] *= scale.y;
            m[1, 2] *= scale.y;
            m[2, 0] *= scale.z;
            m[2, 1] *= scale.z;
            m[2, 2] *= scale.z;

            return m;
        }

		/// <summary>
		/// Transposed dot product with the x axis of the matrix
		/// </summary>
		/// <returns>Transposed dot product</returns>
		public float Tdotx(Vector3 with)
        {
            return this[0, 0] * with[0] + this[1, 0] * with[1] + this[2, 0] * with[2];
        }

		/// <summary>
		/// Transposed dot product with the y axis of the matrix
		/// </summary>
		/// <returns>Transposed dot product</returns>
		public float Tdoty(Vector3 with)
        {
            return this[0, 1] * with[0] + this[1, 1] * with[1] + this[2, 1] * with[2];
        }


		/// <summary>
		/// Transposed dot product with the z axis of the matrix
		/// </summary>
		/// <returns>Transposed dot product</returns>
		public float Tdotz(Vector3 with)
        {
            return this[0, 2] * with[0] + this[1, 2] * with[1] + this[2, 2] * with[2];
        }

		/// <summary>
		/// Transpose the matrix
		/// </summary>
		/// <returns>Transposed version of the matrix</returns>
        public Basis Transposed()
        {
            Basis tr = this;

            float temp = this[0, 1];
            this[0, 1] = this[1, 0];
            this[1, 0] = temp;

            temp = this[0, 2];
            this[0, 2] = this[2, 0];
            this[2, 0] = temp;

            temp = this[1, 2];
            this[1, 2] = this[2, 1];
            this[2, 1] = temp;

            return tr;
        }

		/// <summary>
		/// Transform a vector by the matrix
		/// </summary>
		/// <param name="v">Vector to transform with the matrix</param>
		/// <returns>Transformed vector</returns>
        public Vector3 Xform(Vector3 v)
        {
            return new Vector3
            (
                this[0].dot(v),
                this[1].dot(v),
                this[2].dot(v)
            );
        }

		/// <summary>
		/// Transform a vector by the inverse of matrix
		/// </summary>
		/// <param name="v">Vector to transform with the inverse of the matrix</param>
		/// <remarks>Results in a multiplication by the inverse of the matrix only if it represents a rotation-reflection</remarks>
		/// <returns>Transformed vector</returns>
		public Vector3 XformInv(Vector3 v)
        {
            return new Vector3
            (
                (this[0, 0] * v.x) + (this[1, 0] * v.y) + (this[2, 0] * v.z),
                (this[0, 1] * v.x) + (this[1, 1] * v.y) + (this[2, 1] * v.z),
                (this[0, 2] * v.x) + (this[1, 2] * v.y) + (this[2, 2] * v.z)
            );
        }

		public Quat Quat() {
			float trace = x[0] + y[1] + z[2];

			if (trace > 0.0f) {
				float s = Mathf.sqrt(trace + 1.0f) * 2f;
				float inv_s = 1f / s;
				return new Quat(
					(z[1] - y[2]) * inv_s,
					(x[2] - z[0]) * inv_s,
					(y[0] - x[1]) * inv_s,
					s * 0.25f
				);
			} else if (x[0] > y[1] && x[0] > z[2]) {
				float s = Mathf.sqrt(x[0] - y[1] - z[2] + 1.0f) * 2f;
				float inv_s = 1f / s;
				return new Quat(
					s * 0.25f,
					(x[1] + y[0]) * inv_s,
					(x[2] + z[0]) * inv_s,
					(z[1] - y[2]) * inv_s
				);
			} else if (y[1] > z[2]) {
				float s = Mathf.sqrt(-x[0] + y[1] - z[2] + 1.0f) * 2f;
				float inv_s = 1f / s;
				return new Quat(
					(x[1] + y[0]) * inv_s,
					s * 0.25f,
					(y[2] + z[1]) * inv_s,
					(x[2] - z[0]) * inv_s
				);
			} else {
				float s = Mathf.sqrt(-x[0] - y[1] + z[2] + 1.0f) * 2f;
				float inv_s = 1f / s;
				return new Quat(
					(x[2] + z[0]) * inv_s,
					(y[2] + z[1]) * inv_s,
					s * 0.25f,
					(y[0] - x[1]) * inv_s
				);
			}
		}

        public Basis(Quat quat)
        {
            float s = 2.0f / quat.length_squared();

            float xs = quat.x * s;
            float ys = quat.y * s;
            float zs = quat.z * s;
            float wx = quat.w * xs;
            float wy = quat.w * ys;
            float wz = quat.w * zs;
            float xx = quat.x * xs;
            float xy = quat.x * ys;
            float xz = quat.x * zs;
            float yy = quat.y * ys;
            float yz = quat.y * zs;
            float zz = quat.z * zs;

            this.x = new Vector3(1.0f - (yy + zz), xy - wz, xz + wy);
            this.y = new Vector3(xy + wz, 1.0f - (xx + zz), yz - wx);
            this.z = new Vector3(xz - wy, yz + wx, 1.0f - (xx + yy));
        }

        public Basis(Vector3 axis, float phi)
        {
            Vector3 axis_sq = new Vector3(axis.x * axis.x, axis.y * axis.y, axis.z * axis.z);

            float cosine = Mathf.cos(phi);
            float sine = Mathf.sin(phi);

            this.x = new Vector3
            (
                axis_sq.x + cosine * (1.0f - axis_sq.x),
                axis.x * axis.y * (1.0f - cosine) - axis.z * sine,
                axis.z * axis.x * (1.0f - cosine) + axis.y * sine
            );

            this.y = new Vector3
            (
                axis.x * axis.y * (1.0f - cosine) + axis.z * sine,
                axis_sq.y + cosine * (1.0f - axis_sq.y),
                axis.y * axis.z * (1.0f - cosine) - axis.x * sine
            );

            this.z = new Vector3
            (
                axis.z * axis.x * (1.0f - cosine) - axis.y * sine,
                axis.y * axis.z * (1.0f - cosine) + axis.x * sine,
                axis_sq.z + cosine * (1.0f - axis_sq.z)
            );
        }

        public Basis(Vector3 xAxis, Vector3 yAxis, Vector3 zAxis)
        {
            this.x = xAxis;
            this.y = yAxis;
            this.z = zAxis;
        }

        public Basis(float xx, float xy, float xz, float yx, float yy, float yz, float zx, float zy, float zz)
        {
            this.x = new Vector3(xx, xy, xz);
            this.y = new Vector3(yx, yy, yz);
            this.z = new Vector3(zx, zy, zz);
        }

        public static Basis operator *(Basis left, Basis right)
        {
            return new Basis
            (
                right.Tdotx(left[0]), right.Tdoty(left[0]), right.Tdotz(left[0]),
                right.Tdotx(left[1]), right.Tdoty(left[1]), right.Tdotz(left[1]),
                right.Tdotx(left[2]), right.Tdoty(left[2]), right.Tdotz(left[2])
            );
        }

        public static bool operator ==(Basis left, Basis right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Basis left, Basis right)
        {
            return !left.Equals(right);
        }

        public override bool Equals(object obj)
        {
            if (obj is Basis)
            {
                return Equals((Basis)obj);
            }

            return false;
        }

        public bool Equals(Basis other)
        {
            return x.Equals(other.x) && y.Equals(other.y) && z.Equals(other.z);
        }

        public override int GetHashCode()
        {
            return x.GetHashCode() ^ y.GetHashCode() ^ z.GetHashCode();
        }

        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(),
                this.y.ToString(),
                this.z.ToString()
            });
        }

        public string ToString(string format)
        {
            return String.Format("({0}, {1}, {2})", new object[]
            {
                this.x.ToString(format),
                this.y.ToString(format),
                this.z.ToString(format)
            });
        }
    }
}
