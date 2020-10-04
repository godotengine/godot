#ifndef TRANSFORM2D_H
#define TRANSFORM2D_H

#include "Vector2.hpp"

namespace godot {

typedef Vector2 Size2;

struct Rect2;

struct Transform2D {
	// Warning #1: basis of Transform2D is stored differently from Basis. In terms of elements array, the basis matrix looks like "on paper":
	// M = (elements[0][0] elements[1][0])
	//     (elements[0][1] elements[1][1])
	// This is such that the columns, which can be interpreted as basis vectors of the coordinate system "painted" on the object, can be accessed as elements[i].
	// Note that this is the opposite of the indices in mathematical texts, meaning: $M_{12}$ in a math book corresponds to elements[1][0] here.
	// This requires additional care when working with explicit indices.
	// See https://en.wikipedia.org/wiki/Row-_and_column-major_order for further reading.

	// Warning #2: 2D be aware that unlike 3D code, 2D code uses a left-handed coordinate system: Y-axis points down,
	// and angle is measure from +X to +Y in a clockwise-fashion.

	Vector2 elements[3];

	inline real_t tdotx(const Vector2 &v) const { return elements[0][0] * v.x + elements[1][0] * v.y; }
	inline real_t tdoty(const Vector2 &v) const { return elements[0][1] * v.x + elements[1][1] * v.y; }

	inline const Vector2 &operator[](int p_idx) const { return elements[p_idx]; }
	inline Vector2 &operator[](int p_idx) { return elements[p_idx]; }

	inline Vector2 get_axis(int p_axis) const {
		ERR_FAIL_INDEX_V(p_axis, 3, Vector2());
		return elements[p_axis];
	}
	inline void set_axis(int p_axis, const Vector2 &p_vec) {
		ERR_FAIL_INDEX(p_axis, 3);
		elements[p_axis] = p_vec;
	}

	void invert();
	Transform2D inverse() const;

	void affine_invert();
	Transform2D affine_inverse() const;

	void set_rotation(real_t p_phi);
	real_t get_rotation() const;
	void set_rotation_and_scale(real_t p_phi, const Size2 &p_scale);
	void rotate(real_t p_phi);

	void scale(const Size2 &p_scale);
	void scale_basis(const Size2 &p_scale);
	void translate(real_t p_tx, real_t p_ty);
	void translate(const Vector2 &p_translation);

	real_t basis_determinant() const;

	Size2 get_scale() const;

	inline const Vector2 &get_origin() const { return elements[2]; }
	inline void set_origin(const Vector2 &p_origin) { elements[2] = p_origin; }

	Transform2D scaled(const Size2 &p_scale) const;
	Transform2D basis_scaled(const Size2 &p_scale) const;
	Transform2D translated(const Vector2 &p_offset) const;
	Transform2D rotated(real_t p_phi) const;

	Transform2D untranslated() const;

	void orthonormalize();
	Transform2D orthonormalized() const;

	bool operator==(const Transform2D &p_transform) const;
	bool operator!=(const Transform2D &p_transform) const;

	void operator*=(const Transform2D &p_transform);
	Transform2D operator*(const Transform2D &p_transform) const;

	Transform2D interpolate_with(const Transform2D &p_transform, real_t p_c) const;

	Vector2 basis_xform(const Vector2 &p_vec) const;
	Vector2 basis_xform_inv(const Vector2 &p_vec) const;
	Vector2 xform(const Vector2 &p_vec) const;
	Vector2 xform_inv(const Vector2 &p_vec) const;
	Rect2 xform(const Rect2 &p_vec) const;
	Rect2 xform_inv(const Rect2 &p_vec) const;

	operator String() const;

	Transform2D(real_t xx, real_t xy, real_t yx, real_t yy, real_t ox, real_t oy);

	Transform2D(real_t p_rot, const Vector2 &p_pos);
	inline Transform2D() {
		elements[0][0] = 1.0;
		elements[1][1] = 1.0;
	}
};

} // namespace godot

#endif // TRANSFORM2D_H
