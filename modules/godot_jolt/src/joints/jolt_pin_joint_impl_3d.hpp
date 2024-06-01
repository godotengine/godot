#pragma once

#include "joints/jolt_joint_impl_3d.hpp"

class JoltBodyImpl3D;
class JoltSpace3D;

class JoltPinJointImpl3D final : public JoltJointImpl3D {
public:
	JoltPinJointImpl3D(
		const JoltJointImpl3D& p_old_joint,
		JoltBodyImpl3D* p_body_a,
		JoltBodyImpl3D* p_body_b,
		const Vector3& p_local_a,
		const Vector3& p_local_b
	);

	PhysicsServer3D::JointType get_type() const override { return PhysicsServer3D::JOINT_TYPE_PIN; }

	Vector3 get_local_a() const { return local_ref_a.origin; }

	void set_local_a(const Vector3& p_local_a);

	Vector3 get_local_b() const { return local_ref_b.origin; }

	void set_local_b(const Vector3& p_local_b);

	double get_param(PhysicsServer3D::PinJointParam p_param) const;

	void set_param(PhysicsServer3D::PinJointParam p_param, double p_value);

	float get_applied_force() const;

	void rebuild() override;

private:
	static JPH::Constraint* _build_pin(
		JPH::Body* p_jolt_body_a,
		JPH::Body* p_jolt_body_b,
		const Transform3D& p_shifted_ref_a,
		const Transform3D& p_shifted_ref_b
	);

	void _points_changed();
};
