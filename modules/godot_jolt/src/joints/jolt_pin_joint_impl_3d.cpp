#include "jolt_pin_joint_impl_3d.hpp"

#include "objects/jolt_body_impl_3d.hpp"
#include "spaces/jolt_space_3d.hpp"

namespace {

constexpr double DEFAULT_BIAS = 0.3;
constexpr double DEFAULT_DAMPING = 1.0;
constexpr double DEFAULT_IMPULSE_CLAMP = 0.0;

} // namespace

JoltPinJointImpl3D::JoltPinJointImpl3D(
	const JoltJointImpl3D& p_old_joint,
	JoltBodyImpl3D* p_body_a,
	JoltBodyImpl3D* p_body_b,
	const Vector3& p_local_a,
	const Vector3& p_local_b
)
	: JoltJointImpl3D(
		  p_old_joint,
		  p_body_a,
		  p_body_b,
		  Transform3D({}, p_local_a),
		  Transform3D({}, p_local_b)
	  ) {
	rebuild();
}

void JoltPinJointImpl3D::set_local_a(const Vector3& p_local_a) {
	local_ref_a = Transform3D({}, p_local_a);
	_points_changed();
}

void JoltPinJointImpl3D::set_local_b(const Vector3& p_local_b) {
	local_ref_b = Transform3D({}, p_local_b);
	_points_changed();
}

double JoltPinJointImpl3D::get_param(PhysicsServer3D::PinJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::PIN_JOINT_BIAS: {
			return DEFAULT_BIAS;
		}
		case PhysicsServer3D::PIN_JOINT_DAMPING: {
			return DEFAULT_DAMPING;
		}
		case PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP: {
			return DEFAULT_IMPULSE_CLAMP;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled pin joint parameter: '%d'", p_param));
		}
	}
}

void JoltPinJointImpl3D::set_param(PhysicsServer3D::PinJointParam p_param, double p_value) {
	switch (p_param) {
		case PhysicsServer3D::PIN_JOINT_BIAS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_BIAS)) {
				WARN_PRINT(vformat(
					"Pin joint bias is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::PIN_JOINT_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_DAMPING)) {
				WARN_PRINT(vformat(
					"Pin joint damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::PIN_JOINT_IMPULSE_CLAMP: {
			if (!Math::is_equal_approx(p_value, DEFAULT_IMPULSE_CLAMP)) {
				WARN_PRINT(vformat(
					"Pin joint impulse clamp is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled pin joint parameter: '%d'", p_param));
		} break;
	}
}

float JoltPinJointImpl3D::get_applied_force() const {
	auto* constraint = static_cast<JPH::PointConstraint*>(jolt_ref.GetPtr());
	ERR_FAIL_NULL_D(constraint);

	JoltSpace3D* space = get_space();
	ERR_FAIL_NULL_D(space);

	const float last_step = space->get_last_step();
	QUIET_FAIL_COND_D(last_step == 0.0f);

	return constraint->GetTotalLambdaPosition().Length() / last_step;
}

void JoltPinJointImpl3D::rebuild() {
	destroy();

	JoltSpace3D* space = get_space();

	if (space == nullptr) {
		return;
	}

	const JPH::BodyID body_ids[2] = {
		body_a != nullptr ? body_a->get_jolt_id() : JPH::BodyID(),
		body_b != nullptr ? body_b->get_jolt_id() : JPH::BodyID()};

	const JoltWritableBodies3D jolt_bodies = space->write_bodies(body_ids, count_of(body_ids));

	auto* jolt_body_a = static_cast<JPH::Body*>(jolt_bodies[0]);
	auto* jolt_body_b = static_cast<JPH::Body*>(jolt_bodies[1]);

	ERR_FAIL_COND(jolt_body_a == nullptr && jolt_body_b == nullptr);

	Transform3D shifted_ref_a;
	Transform3D shifted_ref_b;

	_shift_reference_frames(Vector3(), Vector3(), shifted_ref_a, shifted_ref_b);

	jolt_ref = _build_pin(jolt_body_a, jolt_body_b, shifted_ref_a, shifted_ref_b);

	space->add_joint(this);

	_update_enabled();
	_update_iterations();
}

JPH::Constraint* JoltPinJointImpl3D::_build_pin(
	JPH::Body* p_jolt_body_a,
	JPH::Body* p_jolt_body_b,
	const Transform3D& p_shifted_ref_a,
	const Transform3D& p_shifted_ref_b
) {
	JPH::PointConstraintSettings constraint_settings;
	constraint_settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
	constraint_settings.mPoint1 = to_jolt_r(p_shifted_ref_a.origin);
	constraint_settings.mPoint2 = to_jolt_r(p_shifted_ref_b.origin);

	if (p_jolt_body_a == nullptr) {
		return constraint_settings.Create(JPH::Body::sFixedToWorld, *p_jolt_body_b);
	} else if (p_jolt_body_b == nullptr) {
		return constraint_settings.Create(*p_jolt_body_a, JPH::Body::sFixedToWorld);
	} else {
		return constraint_settings.Create(*p_jolt_body_a, *p_jolt_body_b);
	}
}

void JoltPinJointImpl3D::_points_changed() {
	rebuild();
}
