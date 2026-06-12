/**************************************************************************/
/*  test_generic_6dof_joint_3d.cpp                                        */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_generic_6dof_joint_3d)

#ifndef PHYSICS_3D_DISABLED

#include "scene/3d/physics/joints/generic_6dof_joint_3d.h"

namespace TestGeneric6DOFJoint3D {

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Quaternion angular target is normalized and stored") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	const Quaternion target_rotation(Vector3(0, 1, 0), 0.5);
	joint->set_angular_target_rotation(target_rotation * 2.0);

	CHECK(joint->get_angular_target_rotation().is_normalized());
	CHECK(joint->get_angular_target_rotation().is_equal_approx(target_rotation));
	CHECK(joint->has_target_rotation());

	memdelete(joint);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Euler equilibrium setter clears stored quaternion target") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	joint->set_angular_target_rotation(Quaternion(Vector3(0, 0, 1), 0.25));
	CHECK(joint->has_target_rotation());

	joint->set_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0.1);
	CHECK_FALSE(joint->has_target_rotation());

	memdelete(joint);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] clear_angular_target_rotation resets state") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	joint->set_angular_target_rotation(Quaternion(Vector3(1, 0, 0), 0.5));
	CHECK(joint->has_target_rotation());

	joint->clear_angular_target_rotation();
	CHECK_FALSE(joint->has_target_rotation());
	ERR_PRINT_OFF;
	CHECK(joint->get_angular_target_rotation().is_equal_approx(Quaternion()));
	ERR_PRINT_ON;

	memdelete(joint);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Getter returns identity before configuration when no explicit target is set") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	joint->set_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0.3);
	joint->set_param_y(Generic6DOFJoint3D::PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, -0.2);
	joint->set_param_z(Generic6DOFJoint3D::PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0.4);

	CHECK_FALSE(joint->has_target_rotation());
	ERR_PRINT_OFF;
	CHECK(joint->get_angular_target_rotation().is_equal_approx(Quaternion()));
	ERR_PRINT_ON;

	memdelete(joint);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Explicit quaternion target shadows Euler equilibrium derivation") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	joint->set_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_SPRING_EQUILIBRIUM_POINT, 0.3);
	const Quaternion explicit_target(Vector3(0, 1, 0), 0.7);
	joint->set_angular_target_rotation(explicit_target);

	CHECK(joint->has_target_rotation());
	CHECK(joint->get_angular_target_rotation().is_equal_approx(explicit_target));

	memdelete(joint);
}

} // namespace TestGeneric6DOFJoint3D

#endif // PHYSICS_3D_DISABLED
