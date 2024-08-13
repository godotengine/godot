#include "jolt_slider_joint_impl_3d.hpp"

#include "objects/jolt_body_impl_3d.hpp"
#include "spaces/jolt_space_3d.hpp"

namespace {

constexpr double DEFAULT_LINEAR_LIMIT_SOFTNESS = 1.0;
constexpr double DEFAULT_LINEAR_LIMIT_RESTITUTION = 0.7;
constexpr double DEFAULT_LINEAR_LIMIT_DAMPING = 1.0;

constexpr double DEFAULT_LINEAR_MOTION_SOFTNESS = 1.0;
constexpr double DEFAULT_LINEAR_MOTION_RESTITUTION = 0.7;
constexpr double DEFAULT_LINEAR_MOTION_DAMPING = 0.0;

constexpr double DEFAULT_LINEAR_ORTHO_SOFTNESS = 1.0;
constexpr double DEFAULT_LINEAR_ORTHO_RESTITUTION = 0.7;
constexpr double DEFAULT_LINEAR_ORTHO_DAMPING = 1.0;

constexpr double DEFAULT_ANGULAR_LIMIT_UPPER = 0.0;
constexpr double DEFAULT_ANGULAR_LIMIT_LOWER = 0.0;
constexpr double DEFAULT_ANGULAR_LIMIT_SOFTNESS = 1.0;
constexpr double DEFAULT_ANGULAR_LIMIT_RESTITUTION = 0.7;
constexpr double DEFAULT_ANGULAR_LIMIT_DAMPING = 0.0;

constexpr double DEFAULT_ANGULAR_MOTION_SOFTNESS = 1.0;
constexpr double DEFAULT_ANGULAR_MOTION_RESTITUTION = 0.7;
constexpr double DEFAULT_ANGULAR_MOTION_DAMPING = 1.0;

constexpr double DEFAULT_ANGULAR_ORTHO_SOFTNESS = 1.0;
constexpr double DEFAULT_ANGULAR_ORTHO_RESTITUTION = 0.7;
constexpr double DEFAULT_ANGULAR_ORTHO_DAMPING = 1.0;

} // namespace

JoltSliderJointImpl3D::JoltSliderJointImpl3D(
	const JoltJointImpl3D& p_old_joint,
	JoltBodyImpl3D* p_body_a,
	JoltBodyImpl3D* p_body_b,
	const Transform3D& p_local_ref_a,
	const Transform3D& p_local_ref_b
)
	: JoltJointImpl3D(p_old_joint, p_body_a, p_body_b, p_local_ref_a, p_local_ref_b) {
	rebuild();
}

double JoltSliderJointImpl3D::get_param(PhysicsServer3D::SliderJointParam p_param) const {
	switch (p_param) {
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER: {
			return limit_upper;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER: {
			return limit_lower;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS: {
			return DEFAULT_LINEAR_LIMIT_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION: {
			return DEFAULT_LINEAR_LIMIT_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING: {
			return DEFAULT_LINEAR_LIMIT_DAMPING;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS: {
			return DEFAULT_LINEAR_MOTION_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION: {
			return DEFAULT_LINEAR_MOTION_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_DAMPING: {
			return DEFAULT_LINEAR_MOTION_DAMPING;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS: {
			return DEFAULT_LINEAR_ORTHO_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION: {
			return DEFAULT_LINEAR_ORTHO_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING: {
			return DEFAULT_LINEAR_ORTHO_DAMPING;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER: {
			return DEFAULT_ANGULAR_LIMIT_UPPER;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER: {
			return DEFAULT_ANGULAR_LIMIT_LOWER;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS: {
			return DEFAULT_ANGULAR_LIMIT_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION: {
			return DEFAULT_ANGULAR_LIMIT_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING: {
			return DEFAULT_ANGULAR_LIMIT_DAMPING;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS: {
			return DEFAULT_ANGULAR_MOTION_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION: {
			return DEFAULT_ANGULAR_MOTION_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_DAMPING: {
			return DEFAULT_ANGULAR_MOTION_DAMPING;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS: {
			return DEFAULT_ANGULAR_ORTHO_SOFTNESS;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION: {
			return DEFAULT_ANGULAR_ORTHO_RESTITUTION;
		}
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING: {
			return DEFAULT_ANGULAR_ORTHO_DAMPING;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled slider joint parameter: '%d'", p_param));
		}
	}
}

void JoltSliderJointImpl3D::set_param(PhysicsServer3D::SliderJointParam p_param, double p_value) {
	switch (p_param) {
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_UPPER: {
			limit_upper = p_value;
			_limits_changed();
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_LOWER: {
			limit_lower = p_value;
			_limits_changed();
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_LIMIT_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint linear limit softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_LIMIT_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint linear limit restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_LIMIT_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_LIMIT_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint linear limit damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_MOTION_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint linear motion softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_MOTION_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint linear motion restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_MOTION_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_MOTION_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint linear motion damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_ORTHO_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint linear ortho softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_ORTHO_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint linear ortho restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_LINEAR_ORTHO_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint linear ortho damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_UPPER: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_LIMIT_UPPER)) {
				WARN_PRINT(vformat(
					"Slider joint angular limits are not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"Try using Generic6DOFJoint3D instead. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_LOWER: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_LIMIT_LOWER)) {
				WARN_PRINT(vformat(
					"Slider joint angular limits are not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"Try using Generic6DOFJoint3D instead. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_LIMIT_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint angular limit softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_LIMIT_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint angular limit restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_LIMIT_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_LIMIT_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint angular limit damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_MOTION_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint angular motion softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_MOTION_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint angular motion restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_MOTION_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_MOTION_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint angular motion damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_ORTHO_SOFTNESS)) {
				WARN_PRINT(vformat(
					"Slider joint angular ortho softness is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_ORTHO_RESTITUTION)) {
				WARN_PRINT(vformat(
					"Slider joint angular ortho restitution is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		case PhysicsServer3D::SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING: {
			if (!Math::is_equal_approx(p_value, DEFAULT_ANGULAR_ORTHO_DAMPING)) {
				WARN_PRINT(vformat(
					"Slider joint angular ortho damping is not supported by Godot Jolt. "
					"Any such value will be ignored. "
					"This joint connects %s.",
					_bodies_to_string()
				));
			}
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled slider joint parameter: '%d'", p_param));
		} break;
	}
}

double JoltSliderJointImpl3D::get_jolt_param(JoltParameter p_param) const {
	switch (p_param) {
		case JoltPhysicsServer3D::SLIDER_JOINT_LIMIT_SPRING_FREQUENCY: {
			return limit_spring_frequency;
		}
		case JoltPhysicsServer3D::SLIDER_JOINT_LIMIT_SPRING_DAMPING: {
			return limit_spring_damping;
		}
		case JoltPhysicsServer3D::SLIDER_JOINT_MOTOR_TARGET_VELOCITY: {
			return motor_target_speed;
		}
		case JoltPhysicsServer3D::SLIDER_JOINT_MOTOR_MAX_FORCE: {
			return motor_max_force;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled parameter: '%d'", p_param));
		}
	}
}

void JoltSliderJointImpl3D::set_jolt_param(JoltParameter p_param, double p_value) {
	switch (p_param) {
		case JoltPhysicsServer3D::SLIDER_JOINT_LIMIT_SPRING_FREQUENCY: {
			limit_spring_frequency = p_value;
			_limit_spring_changed();
		} break;
		case JoltPhysicsServer3D::SLIDER_JOINT_LIMIT_SPRING_DAMPING: {
			limit_spring_damping = p_value;
			_limit_spring_changed();
		} break;
		case JoltPhysicsServer3D::SLIDER_JOINT_MOTOR_TARGET_VELOCITY: {
			motor_target_speed = p_value;
			_motor_speed_changed();
		} break;
		case JoltPhysicsServer3D::SLIDER_JOINT_MOTOR_MAX_FORCE: {
			motor_max_force = p_value;
			_motor_limit_changed();
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled parameter: '%d'", p_param));
		}
	}
}

bool JoltSliderJointImpl3D::get_jolt_flag(JoltFlag p_flag) const {
	switch (p_flag) {
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_USE_LIMIT: {
			return limits_enabled;
		}
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_USE_LIMIT_SPRING: {
			return limit_spring_enabled;
		}
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_ENABLE_MOTOR: {
			return motor_enabled;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled flag: '%d'", p_flag));
		}
	}
}

void JoltSliderJointImpl3D::set_jolt_flag(JoltFlag p_flag, bool p_enabled) {
	switch (p_flag) {
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_USE_LIMIT: {
			limits_enabled = p_enabled;
			_limits_changed();
		} break;
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_USE_LIMIT_SPRING: {
			limit_spring_enabled = p_enabled;
			_limit_spring_changed();
		} break;
		case JoltPhysicsServer3D::SLIDER_JOINT_FLAG_ENABLE_MOTOR: {
			motor_enabled = p_enabled;
			_motor_state_changed();
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled flag: '%d'", p_flag));
		} break;
	}
}

float JoltSliderJointImpl3D::get_applied_force() const {
	ERR_FAIL_NULL_D(jolt_ref);

	JoltSpace3D* space = get_space();
	ERR_FAIL_NULL_D(space);

	const float last_step = space->get_last_step();
	QUIET_FAIL_COND_D(last_step == 0.0f);

	if (_is_fixed()) {
		auto* constraint = static_cast<JPH::FixedConstraint*>(jolt_ref.GetPtr());
		return constraint->GetTotalLambdaPosition().Length() / last_step;
	} else {
		auto* constraint = static_cast<JPH::SliderConstraint*>(jolt_ref.GetPtr());
		return constraint->GetTotalLambdaPosition().Length() / last_step;
	}
}

float JoltSliderJointImpl3D::get_applied_torque() const {
	ERR_FAIL_NULL_D(jolt_ref);

	JoltSpace3D* space = get_space();
	ERR_FAIL_NULL_D(space);

	const float last_step = space->get_last_step();
	QUIET_FAIL_COND_D(last_step == 0.0f);

	if (_is_fixed()) {
		auto* constraint = static_cast<JPH::FixedConstraint*>(jolt_ref.GetPtr());
		return constraint->GetTotalLambdaRotation().Length() / last_step;
	} else {
		auto* constraint = static_cast<JPH::SliderConstraint*>(jolt_ref.GetPtr());
		return constraint->GetTotalLambdaRotation().Length() / last_step;
	}
}

void JoltSliderJointImpl3D::rebuild() {
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

	float ref_shift = 0.0f;
	float limit = FLT_MAX;

	if (limits_enabled && limit_lower <= limit_upper) {
		const double limit_midpoint = (limit_lower + limit_upper) / 2.0f;

		ref_shift = float(-limit_midpoint);
		limit = float(limit_upper - limit_midpoint);
	}

	Transform3D shifted_ref_a;
	Transform3D shifted_ref_b;

	_shift_reference_frames(
		Vector3(ref_shift, 0.0f, 0.0f),
		Vector3(),
		shifted_ref_a,
		shifted_ref_b
	);

	if (_is_fixed()) {
		jolt_ref = _build_fixed(jolt_body_a, jolt_body_b, shifted_ref_a, shifted_ref_b);
	} else {
		jolt_ref = _build_slider(jolt_body_a, jolt_body_b, shifted_ref_a, shifted_ref_b, limit);
	}

	space->add_joint(this);

	_update_enabled();
	_update_iterations();
	_update_motor_state();
	_update_motor_velocity();
	_update_motor_limit();
}

JPH::Constraint* JoltSliderJointImpl3D::_build_slider(
	JPH::Body* p_jolt_body_a,
	JPH::Body* p_jolt_body_b,
	const Transform3D& p_shifted_ref_a,
	const Transform3D& p_shifted_ref_b,
	float p_limit
) const {
	JPH::SliderConstraintSettings constraint_settings;

	constraint_settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
	constraint_settings.mAutoDetectPoint = false;
	constraint_settings.mPoint1 = to_jolt_r(p_shifted_ref_a.origin);
	constraint_settings.mSliderAxis1 = to_jolt(p_shifted_ref_a.basis.get_column(Vector3::AXIS_X));
	constraint_settings.mNormalAxis1 = to_jolt(p_shifted_ref_a.basis.get_column(Vector3::AXIS_Z));
	constraint_settings.mPoint2 = to_jolt_r(p_shifted_ref_b.origin);
	constraint_settings.mSliderAxis2 = to_jolt(p_shifted_ref_b.basis.get_column(Vector3::AXIS_X));
	constraint_settings.mNormalAxis2 = to_jolt(p_shifted_ref_b.basis.get_column(Vector3::AXIS_Z));
	constraint_settings.mLimitsMin = -p_limit;
	constraint_settings.mLimitsMax = p_limit;

	if (limit_spring_enabled) {
		constraint_settings.mLimitsSpringSettings.mFrequency = (float)limit_spring_frequency;
		constraint_settings.mLimitsSpringSettings.mDamping = (float)limit_spring_damping;
	}

	if (p_jolt_body_a == nullptr) {
		return constraint_settings.Create(JPH::Body::sFixedToWorld, *p_jolt_body_b);
	} else if (p_jolt_body_b == nullptr) {
		return constraint_settings.Create(*p_jolt_body_a, JPH::Body::sFixedToWorld);
	} else {
		return constraint_settings.Create(*p_jolt_body_a, *p_jolt_body_b);
	}
}

JPH::Constraint* JoltSliderJointImpl3D::_build_fixed(
	JPH::Body* p_jolt_body_a,
	JPH::Body* p_jolt_body_b,
	const Transform3D& p_shifted_ref_a,
	const Transform3D& p_shifted_ref_b
) const {
	JPH::FixedConstraintSettings constraint_settings;

	constraint_settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
	constraint_settings.mAutoDetectPoint = false;
	constraint_settings.mPoint1 = to_jolt_r(p_shifted_ref_a.origin);
	constraint_settings.mAxisX1 = to_jolt(p_shifted_ref_a.basis.get_column(Vector3::AXIS_X));
	constraint_settings.mAxisY1 = to_jolt(p_shifted_ref_a.basis.get_column(Vector3::AXIS_Y));
	constraint_settings.mPoint2 = to_jolt_r(p_shifted_ref_b.origin);
	constraint_settings.mAxisX2 = to_jolt(p_shifted_ref_b.basis.get_column(Vector3::AXIS_X));
	constraint_settings.mAxisY2 = to_jolt(p_shifted_ref_b.basis.get_column(Vector3::AXIS_Y));

	if (p_jolt_body_a == nullptr) {
		return constraint_settings.Create(JPH::Body::sFixedToWorld, *p_jolt_body_b);
	} else if (p_jolt_body_b == nullptr) {
		return constraint_settings.Create(*p_jolt_body_a, JPH::Body::sFixedToWorld);
	} else {
		return constraint_settings.Create(*p_jolt_body_a, *p_jolt_body_b);
	}
}

void JoltSliderJointImpl3D::_update_motor_state() {
	QUIET_FAIL_COND(_is_fixed());

	if (auto* constraint = static_cast<JPH::SliderConstraint*>(jolt_ref.GetPtr())) {
		constraint->SetMotorState(
			motor_enabled ? JPH::EMotorState::Velocity : JPH::EMotorState::Off
		);
	}
}

void JoltSliderJointImpl3D::_update_motor_velocity() {
	QUIET_FAIL_COND(_is_fixed());

	if (auto* constraint = static_cast<JPH::SliderConstraint*>(jolt_ref.GetPtr())) {
		constraint->SetTargetVelocity((float)motor_target_speed);
	}
}

void JoltSliderJointImpl3D::_update_motor_limit() {
	QUIET_FAIL_COND(_is_fixed());

	if (auto* constraint = static_cast<JPH::SliderConstraint*>(jolt_ref.GetPtr())) {
		JPH::MotorSettings& motor_settings = constraint->GetMotorSettings();
		motor_settings.mMinForceLimit = (float)-motor_max_force;
		motor_settings.mMaxForceLimit = (float)motor_max_force;
	}
}

void JoltSliderJointImpl3D::_limits_changed() {
	rebuild();
	_wake_up_bodies();
}

void JoltSliderJointImpl3D::_limit_spring_changed() {
	rebuild();
	_wake_up_bodies();
}

void JoltSliderJointImpl3D::_motor_state_changed() {
	_update_motor_state();
}

void JoltSliderJointImpl3D::_motor_speed_changed() {
	_update_motor_velocity();
	_wake_up_bodies();
}

void JoltSliderJointImpl3D::_motor_limit_changed() {
	_update_motor_limit();
	_wake_up_bodies();
}
