/**************************************************************************/
/*  transform3d.hpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "basis.hpp"
#include "variant.hpp"

struct Transform3D {
	constexpr Transform3D() {}

	/// @brief Create a new identity transform.
	/// @return The identity transform.
	static Transform3D identity();

	/// @brief Create a new transform from a basis and origin.
	/// @param origin The origin of the transform.
	/// @param basis The basis of the transform.
	Transform3D(const Vector3 &origin, const Basis &basis);

	Transform3D &operator=(const Transform3D &transform);
	void assign(const Transform3D &transform);

	// Transform3D operations
	void invert();
	void affine_invert();
	void translate(const Vector3 &offset);
	void rotate(const Vector3 &axis, double angle);
	void scale(const Vector3 &scale);

	Transform3D inverse() const;
	Transform3D orthonormalized() const;
	Transform3D rotated(const Vector3 &axis, double angle) const;
	Transform3D rotated_local(const Vector3 &axis, double angle) const;
	Transform3D scaled(const Vector3 &scale) const;
	Transform3D scaled_local(const Vector3 &scale) const;
	Transform3D translated(const Vector3 &offset) const;
	Transform3D translated_local(const Vector3 &offset) const;
	Transform3D looking_at(const Vector3 &target, const Vector3 &up) const;
	Transform3D interpolate_with(const Transform3D &to, double weight) const;

	// Transform3D access
	Vector3 get_origin() const;
	void set_origin(const Vector3 &origin);
	Basis get_basis() const;
	void set_basis(const Basis &basis);

	template <typename... Args>
	Variant operator()(std::string_view method, Args &&...args);

	METHOD(Transform3D, affine_inverse);
	METHOD(bool, is_equal_approx);
	METHOD(bool, is_finite);

	static Transform3D from_variant_index(unsigned idx) {
		Transform3D a{};
		a.m_idx = idx;
		return a;
	}
	unsigned get_variant_index() const noexcept { return m_idx; }

private:
	unsigned m_idx = INT32_MIN;
};

inline Variant::Variant(const Transform3D &t) {
	m_type = Variant::TRANSFORM3D;
	v.i = t.get_variant_index();
}

inline Variant::operator Transform3D() const {
	if (m_type != Variant::TRANSFORM3D) {
		api_throw("std::bad_cast", "Failed to cast Variant to Transform3D", this);
	}
	return Transform3D::from_variant_index(v.i);
}

inline Transform3D Variant::as_transform3d() const {
	return static_cast<Transform3D>(*this);
}

inline Transform3D &Transform3D::operator=(const Transform3D &transform) {
	if (this->m_idx != INT32_MIN) {
		this->assign(transform);
	} else {
		this->m_idx = transform.m_idx;
	}
	return *this;
}

template <typename... Args>
Variant Transform3D::operator()(std::string_view method, Args &&...args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}
