#ifndef BASIS_H
#define BASIS_H

#include <gdnative/basis.h>

#include "Defs.hpp"

#include "Vector3.hpp"

namespace godot {

class Quat;

class Basis {
private:
	// This helper template is for mimicking the behavior difference between the engine
	// and script interfaces that logically script sees matrices as column major, while
	// the engine stores them in row major to efficiently take advantage of SIMD
	// instructions in case of matrix-vector multiplications.
	// With this helper template native scripts see the data as if it was column major
	// without actually transposing the basis matrix at the script-engine boundary.
	template <int column>
	class ColumnVector3 {
	private:
		template <int column1, int component>
		class ColumnVectorComponent {
		private:
			Vector3 elements[3];

		protected:
			inline ColumnVectorComponent<column1, component> &operator=(const ColumnVectorComponent<column1, component> &p_value) {
				return *this = real_t(p_value);
			}

			inline ColumnVectorComponent(const ColumnVectorComponent<column1, component> &p_value) {
				*this = real_t(p_value);
			}

			inline ColumnVectorComponent<column1, component> &operator=(const real_t &p_value) {
				elements[component][column1] = p_value;
				return *this;
			}

			inline operator real_t() const {
				return elements[component][column1];
			}
		};

	public:
		enum Axis {
			AXIS_X,
			AXIS_Y,
			AXIS_Z,
		};

		union {
			ColumnVectorComponent<column, 0> x;
			ColumnVectorComponent<column, 1> y;
			ColumnVectorComponent<column, 2> z;

			Vector3 elements[3]; // Not for direct access, use [] operator instead
		};

		inline ColumnVector3<column> &operator=(const ColumnVector3<column> &p_value) {
			return *this = Vector3(p_value);
		}

		inline ColumnVector3(const ColumnVector3<column> &p_value) {
			*this = Vector3(p_value);
		}

		inline ColumnVector3<column> &operator=(const Vector3 &p_value) {
			elements[0][column] = p_value.x;
			elements[1][column] = p_value.y;
			elements[2][column] = p_value.z;
			return *this;
		}

		inline operator Vector3() const {
			return Vector3(elements[0][column], elements[1][column], elements[2][column]);
		}

		// Unfortunately, we also need to replicate the other interfaces of Vector3 in
		// order for being able to directly operate on these "meta-Vector3" objects without
		// an explicit cast or an intermediate assignment to a real Vector3 object.

		inline const real_t &operator[](int p_axis) const {
			return elements[p_axis][column];
		}

		inline real_t &operator[](int p_axis) {
			return elements[p_axis][column];
		}

		inline ColumnVector3<column> &operator+=(const Vector3 &p_v) {
			return *this = *this + p_v;
		}

		inline Vector3 operator+(const Vector3 &p_v) const {
			return Vector3(*this) + p_v;
		}

		inline ColumnVector3<column> &operator-=(const Vector3 &p_v) {
			return *this = *this - p_v;
		}

		inline Vector3 operator-(const Vector3 &p_v) const {
			return Vector3(*this) - p_v;
		}

		inline ColumnVector3<column> &operator*=(const Vector3 &p_v) {
			return *this = *this * p_v;
		}

		inline Vector3 operator*(const Vector3 &p_v) const {
			return Vector3(*this) * p_v;
		}

		inline ColumnVector3<column> &operator/=(const Vector3 &p_v) {
			return *this = *this / p_v;
		}

		inline Vector3 operator/(const Vector3 &p_v) const {
			return Vector3(*this) / p_v;
		}

		inline ColumnVector3<column> &operator*=(real_t p_scalar) {
			return *this = *this * p_scalar;
		}

		inline Vector3 operator*(real_t p_scalar) const {
			return Vector3(*this) * p_scalar;
		}

		inline ColumnVector3<column> &operator/=(real_t p_scalar) {
			return *this = *this / p_scalar;
		}

		inline Vector3 operator/(real_t p_scalar) const {
			return Vector3(*this) / p_scalar;
		}

		inline Vector3 operator-() const {
			return -Vector3(*this);
		}

		inline bool operator==(const Vector3 &p_v) const {
			return Vector3(*this) == p_v;
		}

		inline bool operator!=(const Vector3 &p_v) const {
			return Vector3(*this) != p_v;
		}

		inline bool operator<(const Vector3 &p_v) const {
			return Vector3(*this) < p_v;
		}

		inline bool operator<=(const Vector3 &p_v) const {
			return Vector3(*this) <= p_v;
		}

		inline Vector3 abs() const {
			return Vector3(*this).abs();
		}

		inline Vector3 ceil() const {
			return Vector3(*this).ceil();
		}

		inline Vector3 cross(const Vector3 &b) const {
			return Vector3(*this).cross(b);
		}

		inline Vector3 linear_interpolate(const Vector3 &p_b, real_t p_t) const {
			return Vector3(*this).linear_interpolate(p_b, p_t);
		}

		inline Vector3 cubic_interpolate(const Vector3 &b, const Vector3 &pre_a, const Vector3 &post_b, const real_t t) const {
			return Vector3(*this).cubic_interpolate(b, pre_a, post_b, t);
		}

		inline Vector3 bounce(const Vector3 &p_normal) const {
			return Vector3(*this).bounce(p_normal);
		}

		inline real_t length() const {
			return Vector3(*this).length();
		}

		inline real_t length_squared() const {
			return Vector3(*this).length_squared();
		}

		inline real_t distance_squared_to(const Vector3 &b) const {
			return Vector3(*this).distance_squared_to(b);
		}

		inline real_t distance_to(const Vector3 &b) const {
			return Vector3(*this).distance_to(b);
		}

		inline real_t dot(const Vector3 &b) const {
			return Vector3(*this).dot(b);
		}

		inline real_t angle_to(const Vector3 &b) const {
			return Vector3(*this).angle_to(b);
		}

		inline Vector3 floor() const {
			return Vector3(*this).floor();
		}

		inline Vector3 inverse() const {
			return Vector3(*this).inverse();
		}

		inline bool is_normalized() const {
			return Vector3(*this).is_normalized();
		}

		inline Basis outer(const Vector3 &b) const {
			return Vector3(*this).outer(b);
		}

		inline int max_axis() const {
			return Vector3(*this).max_axis();
		}

		inline int min_axis() const {
			return Vector3(*this).min_axis();
		}

		inline void normalize() {
			Vector3 v = *this;
			v.normalize();
			*this = v;
		}

		inline Vector3 normalized() const {
			return Vector3(*this).normalized();
		}

		inline Vector3 reflect(const Vector3 &by) const {
			return Vector3(*this).reflect(by);
		}

		inline Vector3 rotated(const Vector3 &axis, const real_t phi) const {
			return Vector3(*this).rotated(axis, phi);
		}

		inline void rotate(const Vector3 &p_axis, real_t p_phi) {
			Vector3 v = *this;
			v.rotate(p_axis, p_phi);
			*this = v;
		}

		inline Vector3 slide(const Vector3 &by) const {
			return Vector3(*this).slide(by);
		}

		inline void snap(real_t p_val) {
			Vector3 v = *this;
			v.snap(p_val);
			*this = v;
		}

		inline Vector3 snapped(const float by) {
			return Vector3(*this).snapped(by);
		}

		inline operator String() const {
			return String(Vector3(*this));
		}
	};

public:
	union {
		ColumnVector3<0> x;
		ColumnVector3<1> y;
		ColumnVector3<2> z;

		Vector3 elements[3]; // Not for direct access, use [] operator instead
	};

	inline Basis(const Basis &p_basis) {
		elements[0] = p_basis.elements[0];
		elements[1] = p_basis.elements[1];
		elements[2] = p_basis.elements[2];
	}

	inline Basis &operator=(const Basis &p_basis) {
		elements[0] = p_basis.elements[0];
		elements[1] = p_basis.elements[1];
		elements[2] = p_basis.elements[2];
		return *this;
	}

	Basis(const Quat &p_quat); // euler
	Basis(const Vector3 &p_euler); // euler
	Basis(const Vector3 &p_axis, real_t p_phi);

	Basis(const Vector3 &row0, const Vector3 &row1, const Vector3 &row2);

	Basis(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz);

	Basis();

	const Vector3 operator[](int axis) const {
		return get_axis(axis);
	}

	ColumnVector3<0> &operator[](int axis) {
		// We need to do a little pointer magic to get this to work, because the
		// ColumnVector3 template takes the axis as a template parameter.
		// Don't touch this unless you're sure what you're doing!
		return (reinterpret_cast<Basis *>(reinterpret_cast<real_t *>(this) + axis))->x;
	}

	void invert();

	bool isequal_approx(const Basis &a, const Basis &b) const;

	bool is_orthogonal() const;

	bool is_rotation() const;

	void transpose();

	Basis inverse() const;

	Basis transposed() const;

	real_t determinant() const;

	Vector3 get_axis(int p_axis) const;

	void set_axis(int p_axis, const Vector3 &p_value);

	void rotate(const Vector3 &p_axis, real_t p_phi);

	Basis rotated(const Vector3 &p_axis, real_t p_phi) const;

	void scale(const Vector3 &p_scale);

	Basis scaled(const Vector3 &p_scale) const;

	Vector3 get_scale() const;

	Basis slerp(Basis b, float t) const;

	Vector3 get_euler_xyz() const;
	void set_euler_xyz(const Vector3 &p_euler);
	Vector3 get_euler_yxz() const;
	void set_euler_yxz(const Vector3 &p_euler);

	inline Vector3 get_euler() const { return get_euler_yxz(); }
	inline void set_euler(const Vector3 &p_euler) { set_euler_yxz(p_euler); }

	// transposed dot products
	real_t tdotx(const Vector3 &v) const;
	real_t tdoty(const Vector3 &v) const;
	real_t tdotz(const Vector3 &v) const;

	bool operator==(const Basis &p_matrix) const;

	bool operator!=(const Basis &p_matrix) const;

	Vector3 xform(const Vector3 &p_vector) const;

	Vector3 xform_inv(const Vector3 &p_vector) const;
	void operator*=(const Basis &p_matrix);

	Basis operator*(const Basis &p_matrix) const;

	void operator+=(const Basis &p_matrix);

	Basis operator+(const Basis &p_matrix) const;

	void operator-=(const Basis &p_matrix);

	Basis operator-(const Basis &p_matrix) const;

	void operator*=(real_t p_val);

	Basis operator*(real_t p_val) const;

	int get_orthogonal_index() const; // down below

	void set_orthogonal_index(int p_index); // down below

	operator String() const;

	void get_axis_and_angle(Vector3 &r_axis, real_t &r_angle) const;

	/* create / set */

	void set(real_t xx, real_t xy, real_t xz, real_t yx, real_t yy, real_t yz, real_t zx, real_t zy, real_t zz);

	Vector3 get_column(int i) const;

	Vector3 get_row(int i) const;
	Vector3 get_main_diagonal() const;

	void set_row(int i, const Vector3 &p_row);

	Basis transpose_xform(const Basis &m) const;

	void orthonormalize();

	Basis orthonormalized() const;

	bool is_symmetric() const;

	Basis diagonalize();

	operator Quat() const;
};

} // namespace godot

#endif // BASIS_H
