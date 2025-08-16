#pragma once

#include "variant.hpp"

/**
 * @brief Quaternion wrapper for godot-cpp Quaternion.
 * Implemented by referencing and mutating a host-side Basis Variant.
**/
struct Quaternion {
	constexpr Quaternion() {} // DON'T TOUCH

	static Quaternion identity();
	Quaternion(double p_x, double p_y, double p_z, double p_w);
	Quaternion(const Vector3 &axis, double angle);
	Quaternion(const Vector3 &euler);

	Quaternion &operator =(const Quaternion &quat);
	void assign(const Quaternion &quat);

	// Quaternion operations
	double dot(const Quaternion &q) const;
	double length_squared() const;
	double length() const;
	void normalize();
	Quaternion normalized() const;
	bool is_normalized() const;
	Quaternion inverse() const;
	Quaternion log() const;
	Quaternion exp() const;
	double angle_to(const Quaternion &to) const;

	Quaternion slerp(const Quaternion &to, double t) const;
	Quaternion slerpni(const Quaternion &to, double t) const;
	Quaternion cubic_interpolate(const Quaternion &b, const Quaternion &pre_a, const Quaternion &post_b, double t) const;
	Quaternion cubic_interpolate_in_time(const Quaternion &b, const Quaternion &pre_a, const Quaternion &post_b, double t, double b_t, double pre_a_t, double post_b_t) const;

	Vector3 get_axis() const;
	double get_angle() const;

	void operator*=(const Quaternion &q);
	Quaternion operator*(const Quaternion &q) const;

	// Quaternion access
	static constexpr int size() { return 4; }
	double operator[](int idx) const;

	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);

	METHOD(Quaternion, from_euler);
	METHOD(Vector3,    get_euler);
	METHOD(bool,       is_equal_approx);
	METHOD(bool,       is_finite);
	METHOD(Quaternion, spherical_cubic_interpolate);
	METHOD(Quaternion, spherical_cubic_interpolate_in_time);

	static Quaternion from_variant_index(unsigned idx) { Quaternion a {}; a.m_idx = idx; return a; }
	unsigned get_variant_index() const noexcept { return m_idx; }
private:
	unsigned m_idx = INT32_MIN;
};

inline Variant::Variant(const Quaternion &q) {
	m_type = Variant::QUATERNION;
	v.i = q.get_variant_index();
}

inline Variant::operator Quaternion() const {
	if (m_type != Variant::QUATERNION) {
		api_throw("std::bad_cast", "Failed to cast Variant to Quaternion", this);
	}
	return Quaternion::from_variant_index(v.i);
}

inline Quaternion Variant::as_quaternion() const {
	return static_cast<Quaternion>(*this);
}

inline Quaternion &Quaternion::operator =(const Quaternion &q) {
	if (this->m_idx != INT32_MIN) {
		this->assign(q);
	} else {
		this->m_idx = q.m_idx;
	}
	return *this;
}

template <typename... Args>
inline Variant Quaternion::operator () (std::string_view method, Args&&... args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
