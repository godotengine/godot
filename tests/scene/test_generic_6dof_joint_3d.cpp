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
#include "servers/physics_3d/physics_server_3d.h"

#include <cfloat>

namespace TestGeneric6DOFJoint3D {

static RID make_configured_generic_6dof_joint(PhysicsServer3D *p_server, RID &r_body_a, RID &r_body_b) {
	r_body_a = p_server->body_create();
	r_body_b = p_server->body_create();

	const RID joint = p_server->joint_create();
	p_server->joint_make_generic_6dof(joint, r_body_a, Transform3D(), r_body_b, Transform3D());
	return joint;
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Drive force and torque limits params round-trip") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	joint->set_param_x(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT, 123.0);
	joint->set_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_DRIVE_TORQUE_LIMIT, 45.0);

	CHECK(joint->get_param_x(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT) == 123.0);
	CHECK(joint->get_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_DRIVE_TORQUE_LIMIT) == 45.0);

	memdelete(joint);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Drive force and torque limits defaults are unlimited") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	CHECK(joint->get_param_x(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT) == FLT_MAX);
	CHECK(joint->get_param_y(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT) == FLT_MAX);
	CHECK(joint->get_param_z(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT) == FLT_MAX);

	CHECK(joint->get_param_x(Generic6DOFJoint3D::PARAM_ANGULAR_DRIVE_TORQUE_LIMIT) == FLT_MAX);
	CHECK(joint->get_param_y(Generic6DOFJoint3D::PARAM_ANGULAR_DRIVE_TORQUE_LIMIT) == FLT_MAX);
	CHECK(joint->get_param_z(Generic6DOFJoint3D::PARAM_ANGULAR_DRIVE_TORQUE_LIMIT) == FLT_MAX);

	memdelete(joint);
}

TEST_CASE("[SceneTree][PhysicsServer3D][Generic6DOFJoint3D] Drive force and torque limits params round-trip per axis") {
	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	REQUIRE(physics_server != nullptr);

	RID body_a;
	RID body_b;
	RID joint = make_configured_generic_6dof_joint(physics_server, body_a, body_b);

	constexpr Vector3::Axis axes[] = { Vector3::AXIS_X, Vector3::AXIS_Y, Vector3::AXIS_Z };
	constexpr real_t linear_values[] = { 12.0, 34.0, 56.0 };
	constexpr real_t angular_values[] = { 7.0, 8.0, 9.0 };

	for (int i = 0; i < 3; i++) {
		physics_server->generic_6dof_joint_set_param(joint, axes[i], PhysicsServer3D::G6DOF_JOINT_LINEAR_DRIVE_FORCE_LIMIT, linear_values[i]);
		physics_server->generic_6dof_joint_set_param(joint, axes[i], PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT, angular_values[i]);
	}

	for (int i = 0; i < 3; i++) {
		CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, axes[i], PhysicsServer3D::G6DOF_JOINT_LINEAR_DRIVE_FORCE_LIMIT), doctest::Approx(linear_values[i]));
		CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, axes[i], PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT), doctest::Approx(angular_values[i]));
	}

	physics_server->free_rid(joint);
	physics_server->free_rid(body_a);
	physics_server->free_rid(body_b);
}

TEST_CASE("[SceneTree][PhysicsServer3D][Generic6DOFJoint3D] Drive force and torque limits defaults are unlimited") {
	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	REQUIRE(physics_server != nullptr);

	RID body_a;
	RID body_b;
	RID joint = make_configured_generic_6dof_joint(physics_server, body_a, body_b);

	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_X, PhysicsServer3D::G6DOF_JOINT_LINEAR_DRIVE_FORCE_LIMIT), FLT_MAX);
	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_Y, PhysicsServer3D::G6DOF_JOINT_LINEAR_DRIVE_FORCE_LIMIT), FLT_MAX);
	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_Z, PhysicsServer3D::G6DOF_JOINT_LINEAR_DRIVE_FORCE_LIMIT), FLT_MAX);

	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_X, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT), FLT_MAX);
	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_Y, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT), FLT_MAX);
	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_Z, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT), FLT_MAX);

	physics_server->free_rid(joint);
	physics_server->free_rid(body_a);
	physics_server->free_rid(body_b);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Legacy motor and DRIVE limits are stored independently") {
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);

	// Set legacy motor limit, then set DRIVE limit to a finite value.
	ERR_PRINT_OFF;
	joint->set_param_x(Generic6DOFJoint3D::PARAM_LINEAR_MOTOR_FORCE_LIMIT, 50.0);
	ERR_PRINT_ON;
	joint->set_param_x(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT, 80.0);

	CHECK_EQ(joint->get_param_x(Generic6DOFJoint3D::PARAM_LINEAR_MOTOR_FORCE_LIMIT), doctest::Approx(50.0));
	CHECK_EQ(joint->get_param_x(Generic6DOFJoint3D::PARAM_LINEAR_DRIVE_FORCE_LIMIT), doctest::Approx(80.0));

	memdelete(joint);
}

TEST_CASE("[SceneTree][PhysicsServer3D][Generic6DOFJoint3D] DRIVE FLT_MAX after legacy motor round-trips through storage") {
	PhysicsServer3D *physics_server = PhysicsServer3D::get_singleton();
	REQUIRE(physics_server != nullptr);

	RID body_a;
	RID body_b;
	RID joint = make_configured_generic_6dof_joint(physics_server, body_a, body_b);

	// Legacy motor limit set first, then explicit DRIVE FLT_MAX
	physics_server->generic_6dof_joint_set_param(joint, Vector3::AXIS_X, PhysicsServer3D::G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT, 50.0);
	physics_server->generic_6dof_joint_set_param(joint, Vector3::AXIS_X, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT, FLT_MAX);

	CHECK_EQ(physics_server->generic_6dof_joint_get_param(joint, Vector3::AXIS_X, PhysicsServer3D::G6DOF_JOINT_ANGULAR_DRIVE_TORQUE_LIMIT), FLT_MAX);

	physics_server->free_rid(joint);
	physics_server->free_rid(body_a);
	physics_server->free_rid(body_b);
}

TEST_CASE("[SceneTree][Generic6DOFJoint3D] Default joint configuration does not emit MOTOR_FORCE_LIMIT deprecation warning") {
	// Instancing a node should be silent
	Generic6DOFJoint3D *joint = memnew(Generic6DOFJoint3D);
	CHECK(joint != nullptr);
	memdelete(joint);
}

} // namespace TestGeneric6DOFJoint3D

#endif // PHYSICS_3D_DISABLED
