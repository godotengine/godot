#pragma once
#include "core/math/math_funcs.h"
// NOLINTBEGIN(readability-identifier-naming)

#define Mathf_SQRT12 ((float)Math_SQRT12)
#define Mathf_SQRT2 ((float)Math_SQRT2)
#define Mathf_LN2 ((float)Math_LN2)
#define Mathf_PI ((float)Math_PI)
#define Mathf_TAU ((float)Math_TAU)
#define Mathf_E ((float)Math_E)
#define Mathf_INF ((float)Math_INF)
#define Mathf_NAN ((float)Math_NAN)

// NOLINTEND(readability-identifier-naming)

//#define USEC_TO_SEC(m_usec) (double(m_usec) / 1000000.0)

//namespace godot::Math {

_FORCE_INLINE_ void decompose(Basis& p_basis, Vector3& p_scale) {
	Vector3 x = p_basis.get_column(Vector3::AXIS_X);
	Vector3 y = p_basis.get_column(Vector3::AXIS_Y);
	Vector3 z = p_basis.get_column(Vector3::AXIS_Z);

	const float x_dot_x = x.dot(x);

	y -= x * (y.dot(x) / x_dot_x);
	z -= x * (z.dot(x) / x_dot_x);

	const float y_dot_y = y.dot(y);

	z -= y * (z.dot(y) / y_dot_y);

	const float z_dot_z = z.dot(z);

	p_scale = Vector3(Math::sqrt(x_dot_x), Math::sqrt(y_dot_y), Math::sqrt(z_dot_z));

	p_basis.set_column(Vector3::AXIS_X, x / p_scale.x);
	p_basis.set_column(Vector3::AXIS_Y, y / p_scale.y);
	p_basis.set_column(Vector3::AXIS_Z, z / p_scale.z);
}

_FORCE_INLINE_ void decompose(Transform3D& p_transform, Vector3& p_scale) {
	decompose(p_transform.basis, p_scale);
}

_FORCE_INLINE_ Basis decomposed(Basis p_basis, Vector3& p_scale) {
	decompose(p_basis, p_scale);
	return p_basis;
}

_FORCE_INLINE_ Transform3D decomposed(Transform3D p_transform, Vector3& p_scale) {
	decompose(p_transform, p_scale);
	return p_transform;
}

_FORCE_INLINE_ float square(float p_value) {
	return p_value * p_value;
}

//} // namespace godot::Math
