#pragma once

#include "joints/jolt_joint_3d.hpp"

class JoltPinJoint3D final : public JoltJoint3D {
	GDCLASS(JoltPinJoint3D, JoltJoint3D)

private:
	static void _bind_methods();

public:
	float get_applied_force() const;

private:
	void _configure(PhysicsBody3D* p_body_a, PhysicsBody3D* p_body_b) override;
};
