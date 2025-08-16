#pragma once

#include "variant.hpp"

/**
 * @brief Basis wrapper for godot-cpp Basis.
 * Implemented by referencing and mutating a host-side Basis Variant.
 */
struct Basis {
	constexpr Basis() {} // DON'T TOUCH

	/// @brief Create a new identity basis.
	/// @return The identity basis.
	static Basis identity();

	/// @brief Create a new basis from three axes.
	/// @param x  The x-axis of the basis.
	/// @param y  The y-axis of the basis.
	/// @param z  The z-axis of the basis.
	Basis(const Vector3 &x, const Vector3 &y, const Vector3 &z);

	Basis &operator =(const Basis &basis);
	void assign(const Basis &basis);

	// Basis operations
	void invert();
	void transpose();
	Basis inverse() const;
	Basis transposed() const;
	double determinant() const;

	Basis rotated(const Vector3 &axis, double angle) const;
	Basis lerp(const Basis &to, double t) const;
	Basis slerp(const Basis &to, double t) const;

	// Basis access
	Vector3 operator[](int idx) const { return get_row(idx); }

	void set_row(int idx, const Vector3 &axis);
	Vector3 get_row(int idx) const;
	void set_column(int idx, const Vector3 &axis);
	Vector3 get_column(int idx) const;

	// Basis size
	static constexpr int size() { return 3; }

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Basis,  from_euler);
	METHOD(Basis,  from_scale);
	METHOD(Vector3, get_euler);
	VMETHOD(get_rotation_quaternion);
	METHOD(Vector3, get_scale);
	METHOD(bool,   is_conformal);
	METHOD(bool,   is_equal_approx);
	METHOD(bool,   is_finite);
	METHOD(Basis,  looking_at);
	METHOD(Basis,  orthonormalized);
	METHOD(Basis,  scaled);
	METHOD(real_t, tdotx);
	METHOD(real_t, tdoty);
	METHOD(real_t, tdotz);

	static Basis from_variant_index(unsigned idx) { Basis a {}; a.m_idx = idx; return a; }
	unsigned get_variant_index() const noexcept { return m_idx; }
private:
	unsigned m_idx = INT32_MIN;
};

inline Variant::Variant(const Basis &b) {
	m_type = Variant::BASIS;
	v.i = b.get_variant_index();
}

inline Variant::operator Basis() const {
	if (m_type != Variant::BASIS) {
		api_throw("std::bad_cast", "Failed to cast Variant to Basis", this);
	}
	return Basis::from_variant_index(v.i);
}

inline Basis Variant::as_basis() const {
	return static_cast<Basis>(*this);
}

inline Basis &Basis::operator =(const Basis &basis) {
	if (this->m_idx != INT32_MIN) {
		this->assign(basis);
	} else {
		this->m_idx = basis.m_idx;
	}
	return *this;
}

template <typename... Args>
inline Variant Basis::operator () (std::string_view method, Args&&... args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
