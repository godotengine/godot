/**************************************************************************/
/*  import_utils.cpp                                                      */
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

#include "import_utils.h"

Vector3 ImportUtils::deg2rad(const Vector3 &p_rotation) {
	return p_rotation / 180.0 * Math_PI;
}

Vector3 ImportUtils::rad2deg(const Vector3 &p_rotation) {
	return p_rotation / Math_PI * 180.0;
}

Basis ImportUtils::EulerToBasis(FBXDocParser::Model::RotOrder mode, const Vector3 &p_rotation) {
	Basis ret;

	// FBX is using intrinsic euler, we can convert intrinsic to extrinsic (the one used in godot
	// by simply invert its order: https://www.cs.utexas.edu/~theshark/courses/cs354/lectures/cs354-14.pdf
	switch (mode) {
		case FBXDocParser::Model::RotOrder_EulerXYZ:
			ret.set_euler_zyx(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_EulerXZY:
			ret.set_euler_yzx(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_EulerYZX:
			ret.set_euler_xzy(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_EulerYXZ:
			ret.set_euler_zxy(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_EulerZXY:
			ret.set_euler_yxz(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_EulerZYX:
			ret.set_euler_xyz(p_rotation);
			break;

		case FBXDocParser::Model::RotOrder_SphericXYZ:
			// TODO do this.
			break;

		default:
			// If you land here, Please integrate all enums.
			CRASH_NOW_MSG("This is not unreachable.");
	}

	return ret;
}

Quat ImportUtils::EulerToQuaternion(FBXDocParser::Model::RotOrder mode, const Vector3 &p_rotation) {
	return ImportUtils::EulerToBasis(mode, p_rotation);
}

Vector3 ImportUtils::BasisToEuler(FBXDocParser::Model::RotOrder mode, const Basis &p_rotation) {
	// FBX is using intrinsic euler, we can convert intrinsic to extrinsic (the one used in godot
	// by simply invert its order: https://www.cs.utexas.edu/~theshark/courses/cs354/lectures/cs354-14.pdf
	switch (mode) {
		case FBXDocParser::Model::RotOrder_EulerXYZ:
			return p_rotation.get_euler_zyx();

		case FBXDocParser::Model::RotOrder_EulerXZY:
			return p_rotation.get_euler_yzx();

		case FBXDocParser::Model::RotOrder_EulerYZX:
			return p_rotation.get_euler_xzy();

		case FBXDocParser::Model::RotOrder_EulerYXZ:
			return p_rotation.get_euler_zxy();

		case FBXDocParser::Model::RotOrder_EulerZXY:
			return p_rotation.get_euler_yxz();

		case FBXDocParser::Model::RotOrder_EulerZYX:
			return p_rotation.get_euler_xyz();

		case FBXDocParser::Model::RotOrder_SphericXYZ:
			// TODO
			return Vector3();

		default:
			// If you land here, Please integrate all enums.
			CRASH_NOW_MSG("This is not unreachable.");
			return Vector3();
	}
}

Vector3 ImportUtils::QuaternionToEuler(FBXDocParser::Model::RotOrder mode, const Quat &p_rotation) {
	return BasisToEuler(mode, p_rotation);
}

Transform get_unscaled_transform(const Transform &p_initial, real_t p_scale) {
	Transform unscaled = Transform(p_initial.basis, p_initial.origin * p_scale);
	ERR_FAIL_COND_V_MSG(unscaled.basis.determinant() == 0, Transform(), "det is zero unscaled?");
	return unscaled;
}

Vector3 get_poly_normal(const std::vector<Vector3> &p_vertices) {
	ERR_FAIL_COND_V_MSG(p_vertices.size() < 3, Vector3(0, 0, 0), "At least 3 vertices are necessary");
	// Using long double to make sure that normal is computed for even really tiny objects.
	typedef long double ldouble;
	ldouble x = 0.0;
	ldouble y = 0.0;
	ldouble z = 0.0;
	for (size_t i = 0; i < p_vertices.size(); i += 1) {
		const Vector3 current = p_vertices[i];
		const Vector3 next = p_vertices[(i + 1) % p_vertices.size()];
		x += (ldouble(current.y) - ldouble(next.y)) * (ldouble(current.z) + ldouble(next.z));
		y += (ldouble(current.z) - ldouble(next.z)) * (ldouble(current.x) + ldouble(next.x));
		z += (ldouble(current.x) - ldouble(next.x)) * (ldouble(current.y) + ldouble(next.y));
	}
	const ldouble l2 = x * x + y * y + z * z;
	if (l2 == 0.0) {
		return (p_vertices[0] - p_vertices[1]).normalized().cross((p_vertices[0] - p_vertices[2]).normalized()).normalized();
	} else {
		const double l = Math::sqrt(double(l2));
		return Vector3(x / l, y / l, z / l);
	}
}
