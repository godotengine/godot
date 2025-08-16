#pragma once

#include "variant.hpp"

/**
 * @brief Transform2D wrapper for godot-cpp Transform2D.
 * Implemented by referencing and mutating a host-side Transform2D Variant.
**/
struct Transform2D {
	constexpr Transform2D() {} // DON'T TOUCH

	/// @brief Create a new identity transform.
	/// @return The identity transform.
	static Transform2D identity();

	/// @brief Create a new transform from two axes and an origin.
	/// @param x  The x-axis of the transform.
	/// @param y  The y-axis of the transform.
	/// @param origin The origin of the transform.
	Transform2D(const Vector2 &x, const Vector2 &y, const Vector2 &origin);

	Transform2D &operator =(const Transform2D &transform);
	void assign(const Transform2D &transform);

	// Transform2D operations
	void invert();
	void affine_invert();
	void rotate(const double angle);
	void scale(const Vector2 &scale);
	void translate(const Vector2 &offset);
	void interpolate_with(const Transform2D &transform, double weight);

	Transform2D inverse() const;
	Transform2D orthonormalized() const;
	Transform2D rotated(double angle) const;
	Transform2D scaled(const Vector2 &scale) const;
	Transform2D translated(const Vector2 &offset) const;
	Transform2D interpolate_with(const Transform2D &p_transform, double weight) const;

	// Transform2D access
	Vector2 get_column(int idx) const;
	void set_column(int idx, const Vector2 &axis);
	Vector2 operator[](int idx) const { return get_column(idx); }

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Transform2D, affine_inverse);
	METHOD(Vector2, basis_xform);
	METHOD(Vector2, basis_xform_inv);
	METHOD(real_t,  determinant);
	METHOD(Vector2, get_origin);
	METHOD(real_t,  get_rotation);
	METHOD(Vector2, get_scale);
	METHOD(real_t,  get_skew);
	METHOD(bool,    is_conformal);
	METHOD(bool,    is_equal_approx);
	METHOD(bool,    is_finite);
	METHOD(Transform2D, looking_at);
	METHOD(Transform2D, rotated_local);
	METHOD(Transform2D, scaled_local);
	METHOD(Transform2D, translated_local);

	static Transform2D from_variant_index(unsigned idx) { Transform2D a {}; a.m_idx = idx; return a; }
	unsigned get_variant_index() const noexcept { return m_idx; }
private:
	unsigned m_idx = INT32_MIN;
};

inline Variant::Variant(const Transform2D &t) {
	m_type = Variant::TRANSFORM2D;
	v.i = t.get_variant_index();
}

inline Variant::operator Transform2D() const {
	if (m_type != Variant::TRANSFORM2D) {
		api_throw("std::bad_cast", "Failed to cast Variant to Transform2D", this);
	}
	return Transform2D::from_variant_index(v.i);
}

inline Transform2D Variant::as_transform2d() const {
	return static_cast<Transform2D>(*this);
}

inline Transform2D &Transform2D::operator =(const Transform2D &transform) {
	if (this->m_idx != INT32_MIN) {
		this->assign(transform);
	} else {
		this->m_idx = transform.m_idx;
	}
	return *this;
}

template <typename... Args>
inline Variant Transform2D::operator () (std::string_view method, Args&&... args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
