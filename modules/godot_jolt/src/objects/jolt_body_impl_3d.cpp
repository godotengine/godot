#include "jolt_body_impl_3d.hpp"

#include "joints/jolt_joint_impl_3d.hpp"
#include "objects/jolt_area_impl_3d.hpp"
#include "objects/jolt_group_filter.hpp"
#include "objects/jolt_physics_direct_body_state_3d.hpp"
#include "objects/jolt_soft_body_impl_3d.hpp"
#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_broad_phase_layer.hpp"
#include "spaces/jolt_space_3d.hpp"

namespace {

template<typename TValue, typename TGetter>
bool integrate(TValue& p_value, PhysicsServer3D::AreaSpaceOverrideMode p_mode, TGetter&& p_getter) {
	switch (p_mode) {
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED: {
			return false;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE: {
			p_value += p_getter();
			return false;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE: {
			p_value += p_getter();
			return true;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE: {
			p_value = p_getter();
			return true;
		}
		case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE_COMBINE: {
			p_value = p_getter();
			return false;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled override mode: '%d'", p_mode));
		}
	}
}

} // namespace

JoltBodyImpl3D::JoltBodyImpl3D()
	: JoltShapedObjectImpl3D(OBJECT_TYPE_BODY) { }

JoltBodyImpl3D::~JoltBodyImpl3D() {
	memdelete_safely(direct_state);
}

void JoltBodyImpl3D::set_transform(Transform3D p_transform) {
#ifdef TOOLS_ENABLED
	if (unlikely(p_transform.basis.determinant() == 0.0f)) {
		ERR_PRINT(vformat(
			"Failed to set transform for body '%s'. "
			"Its basis was found to be singular, which is not supported by Godot Jolt. "
			"This is likely caused by one or more axes having a scale of zero. "
			"Its basis (and thus its scale) will be treated as identity.",
			to_string()
		));

		p_transform.basis = Basis();
	}
#endif // TOOLS_ENABLED

	Vector3 new_scale;
	decompose(p_transform, new_scale);

	if (!scale.is_equal_approx(new_scale)) {
		scale = new_scale;
		float s = MAX(scale.x,MAX( scale.y , scale.z));
		
		// jolt 不支持非等比缩放, 因此将其设置为等比缩放,因为非等比缩放加旋转可能导致碰撞模型斜切变形
		scale.x = s;
		scale.y = s;
		scale.z = s;
		//WARN_PRINT(_T("jolt does not support non-equal scale, so it is set to equal scale"));
		
		_shapes_changed();
	}

	if (space == nullptr) {
		jolt_settings->mPosition = to_jolt_r(p_transform.origin);
		jolt_settings->mRotation = to_jolt(p_transform.basis);
	} else if (is_kinematic()) {
		kinematic_transform = p_transform;
	} else {
		space->get_body_iface().SetPositionAndRotation(
			jolt_id,
			to_jolt_r(p_transform.origin),
			to_jolt(p_transform.basis),
			JPH::EActivation::DontActivate
		);
	}

	_transform_changed();
}

Variant JoltBodyImpl3D::get_state(PhysicsServer3D::BodyState p_state) const {
	switch (p_state) {
		case PhysicsServer3D::BODY_STATE_TRANSFORM: {
			return get_transform_scaled();
		}
		case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY: {
			return get_linear_velocity();
		}
		case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY: {
			return get_angular_velocity();
		}
		case PhysicsServer3D::BODY_STATE_SLEEPING: {
			return is_sleeping();
		}
		case PhysicsServer3D::BODY_STATE_CAN_SLEEP: {
			return can_sleep();
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled body state: '%d'", p_state));
		}
	}
}

void JoltBodyImpl3D::set_state(PhysicsServer3D::BodyState p_state, const Variant& p_value) {
	switch (p_state) {
		case PhysicsServer3D::BODY_STATE_TRANSFORM: {
			set_transform(p_value);
		} break;
		case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY: {
			set_linear_velocity(p_value);
		} break;
		case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY: {
			set_angular_velocity(p_value);
		} break;
		case PhysicsServer3D::BODY_STATE_SLEEPING: {
			set_is_sleeping(p_value);
		} break;
		case PhysicsServer3D::BODY_STATE_CAN_SLEEP: {
			set_can_sleep(p_value);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled body state: '%d'", p_state));
		} break;
	}
}

Variant JoltBodyImpl3D::get_param(PhysicsServer3D::BodyParameter p_param) const {
	switch (p_param) {
		case PhysicsServer3D::BODY_PARAM_BOUNCE: {
			return get_bounce();
		}
		case PhysicsServer3D::BODY_PARAM_FRICTION: {
			return get_friction();
		}
		case PhysicsServer3D::BODY_PARAM_MASS: {
			return get_mass();
		}
		case PhysicsServer3D::BODY_PARAM_INERTIA: {
			return get_inertia();
		}
		case PhysicsServer3D::BODY_PARAM_CENTER_OF_MASS: {
			return get_center_of_mass_custom();
		}
		case PhysicsServer3D::BODY_PARAM_GRAVITY_SCALE: {
			return get_gravity_scale();
		}
		case PhysicsServer3D::BODY_PARAM_LINEAR_DAMP_MODE: {
			return get_linear_damp_mode();
		}
		case PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP_MODE: {
			return get_angular_damp_mode();
		}
		case PhysicsServer3D::BODY_PARAM_LINEAR_DAMP: {
			return get_linear_damp();
		}
		case PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP: {
			return get_angular_damp();
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled body parameter: '%d'", p_param));
		}
	}
}

void JoltBodyImpl3D::set_param(PhysicsServer3D::BodyParameter p_param, const Variant& p_value) {
	switch (p_param) {
		case PhysicsServer3D::BODY_PARAM_BOUNCE: {
			set_bounce(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_FRICTION: {
			set_friction(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_MASS: {
			set_mass(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_INERTIA: {
			set_inertia(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_CENTER_OF_MASS: {
			set_center_of_mass_custom(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_GRAVITY_SCALE: {
			set_gravity_scale(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_LINEAR_DAMP_MODE: {
			set_linear_damp_mode((DampMode)(int32_t)p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP_MODE: {
			set_angular_damp_mode((DampMode)(int32_t)p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_LINEAR_DAMP: {
			set_linear_damp(p_value);
		} break;
		case PhysicsServer3D::BODY_PARAM_ANGULAR_DAMP: {
			set_angular_damp(p_value);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled body parameter: '%d'", p_param));
		} break;
	}
}

void JoltBodyImpl3D::set_custom_integrator(bool p_enabled) {
	if (custom_integrator == p_enabled) {
		return;
	}

	ON_SCOPE_EXIT {
		_motion_changed();
	};

	custom_integrator = p_enabled;

	if (space == nullptr) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->ResetForce();
	body->ResetTorque();
}

bool JoltBodyImpl3D::is_sleeping() const {
	if (space == nullptr) {
		// HACK(mihe): Since `BODY_STATE_TRANSFORM` will be set right after creation it's more or
		// less impossible to have a body be sleeping when created, so we simply report this as not
		// sleeping.
		return false;
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return !body->IsActive();
}

void JoltBodyImpl3D::set_is_sleeping(bool p_enabled) {
	if (space == nullptr) {
		// HACK(mihe): Since `BODY_STATE_TRANSFORM` will be set right after creation it's more or
		// less impossible to have a body be sleeping when created, so we don't bother storing this.
		return;
	}

	JPH::BodyInterface& body_iface = space->get_body_iface();

	if (p_enabled) {
		body_iface.DeactivateBody(jolt_id);
	} else {
		body_iface.ActivateBody(jolt_id);
		body_iface.ResetSleepTimer(jolt_id);
	}
}

bool JoltBodyImpl3D::can_sleep() const {
	if (space == nullptr) {
		return jolt_settings->mAllowSleeping;
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return body->GetAllowSleeping();
}

void JoltBodyImpl3D::set_can_sleep(bool p_enabled) {
	if (space == nullptr) {
		jolt_settings->mAllowSleeping = p_enabled;
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->SetAllowSleeping(p_enabled);
}

Basis JoltBodyImpl3D::get_principal_inertia_axes() const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve principal inertia axes of '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	if (is_static() || is_kinematic()) {
		return {};
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return to_godot(body->GetRotation() * body->GetMotionProperties()->GetInertiaRotation());
}

Vector3 JoltBodyImpl3D::get_inverse_inertia() const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve inverse inertia of '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	if (is_static() || is_kinematic()) {
		return {};
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	const JPH::MotionProperties& motion_properties = *body->GetMotionPropertiesUnchecked();

	return to_godot(motion_properties.GetLocalSpaceInverseInertia().GetDiagonal3());
}

Basis JoltBodyImpl3D::get_inverse_inertia_tensor() const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve inverse inertia tensor of '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	if (is_static() || is_kinematic()) {
		return {};
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return to_godot(body->GetInverseInertia()).basis;
}

void JoltBodyImpl3D::set_linear_velocity(const Vector3& p_velocity) {
	ON_SCOPE_EXIT {
		_motion_changed();
	};

	if (is_static() || is_kinematic()) {
		linear_surface_velocity = p_velocity;
		return;
	}

	if (space == nullptr) {
		jolt_settings->mLinearVelocity = to_jolt(p_velocity);
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetMotionPropertiesUnchecked()->SetLinearVelocityClamped(to_jolt(p_velocity));
}

void JoltBodyImpl3D::set_angular_velocity(const Vector3& p_velocity) {
	ON_SCOPE_EXIT {
		_motion_changed();
	};

	if (is_static() || is_kinematic()) {
		angular_surface_velocity = p_velocity;
		return;
	}

	if (space == nullptr) {
		jolt_settings->mAngularVelocity = to_jolt(p_velocity);
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetMotionPropertiesUnchecked()->SetAngularVelocityClamped(to_jolt(p_velocity));
}

void JoltBodyImpl3D::set_axis_velocity(const Vector3& p_axis_velocity) {
	ON_SCOPE_EXIT {
		_motion_changed();
	};

	const Vector3 axis = p_axis_velocity.normalized();

	if (space == nullptr) {
		Vector3 linear_velocity = to_godot(jolt_settings->mLinearVelocity);
		linear_velocity -= axis * axis.dot(linear_velocity);
		linear_velocity += p_axis_velocity;
		jolt_settings->mLinearVelocity = to_jolt(linear_velocity);
	} else {
		const JoltWritableBody3D body = space->write_body(jolt_id);
		ERR_FAIL_COND(body.is_invalid());

		Vector3 linear_velocity = get_linear_velocity();
		linear_velocity -= axis * axis.dot(linear_velocity);
		linear_velocity += p_axis_velocity;
		set_linear_velocity(linear_velocity);
	}
}

Vector3 JoltBodyImpl3D::get_velocity_at_position(const Vector3& p_position) const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve point velocity for '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	const JPH::MotionProperties& motion_properties = *body->GetMotionPropertiesUnchecked();

	const Vector3 total_linear_velocity = to_godot(motion_properties.GetLinearVelocity()) +
		linear_surface_velocity;

	const Vector3 total_angular_velocity = to_godot(motion_properties.GetAngularVelocity()) +
		angular_surface_velocity;

	const Vector3 com_to_pos = p_position - to_godot(body->GetCenterOfMassPosition());

	return total_linear_velocity + total_angular_velocity.cross(com_to_pos);
}

void JoltBodyImpl3D::set_center_of_mass_custom(const Vector3& p_center_of_mass) {
	if (custom_center_of_mass && p_center_of_mass == center_of_mass_custom) {
		return;
	}

	custom_center_of_mass = true;
	center_of_mass_custom = p_center_of_mass;

	_shapes_changed();
}

void JoltBodyImpl3D::set_max_contacts_reported(int32_t p_count) {
	ERR_FAIL_COND(p_count < 0);
	QUIET_FAIL_COND(contacts.size() == p_count);

	ON_SCOPE_EXIT {
		_contact_reporting_changed();
	};

	contacts.resize(p_count);
	contact_count = MIN(contact_count, p_count);

	const bool use_manifold_reduction = !reports_contacts();

	if (space == nullptr) {
		jolt_settings->mUseManifoldReduction = use_manifold_reduction;
		return;
	}

	JPH::BodyInterface& body_iface = space->get_body_iface();

	body_iface.SetUseManifoldReduction(jolt_id, use_manifold_reduction);
}

bool JoltBodyImpl3D::reports_all_kinematic_contacts() const {
	return reports_contacts() && JoltProjectSettings::report_all_kinematic_contacts();
}

void JoltBodyImpl3D::add_contact(
	const JoltBodyImpl3D* p_collider,
	float p_depth,
	int32_t p_shape_index,
	int32_t p_collider_shape_index,
	const Vector3& p_normal,
	const Vector3& p_position,
	const Vector3& p_collider_position,
	const Vector3& p_velocity,
	const Vector3& p_collider_velocity,
	const Vector3& p_impulse
) {
	const int32_t max_contacts = get_max_contacts_reported();

	if (max_contacts == 0) {
		return;
	}

	Contact* contact = nullptr;

	if (contact_count < max_contacts) {
		contact = &contacts[contact_count++];
	} else {
		auto shallowest = std::min_element(
			contacts.begin(),
			contacts.end(),
			[](const Contact& p_lhs, const Contact& p_rhs) {
				return p_lhs.depth < p_rhs.depth;
			}
		);

		if (shallowest->depth < p_depth) {
			contact = &*shallowest;
		}
	}

	if (contact != nullptr) {
		contact->shape_index = p_shape_index;
		contact->collider_shape_index = p_collider_shape_index;
		contact->collider_id = p_collider->get_instance_id();
		contact->collider_rid = p_collider->get_rid();
		contact->normal = p_normal;
		contact->position = p_position;
		contact->collider_position = p_collider_position;
		contact->velocity = p_velocity;
		contact->collider_velocity = p_collider_velocity;
		contact->impulse = p_impulse;
	}
}

void JoltBodyImpl3D::reset_mass_properties() {
	if (custom_center_of_mass) {
		custom_center_of_mass = false;
		center_of_mass_custom.zero();

		_shapes_changed();
	}

	inertia.zero();

	_update_mass_properties();
}

void JoltBodyImpl3D::apply_force(const Vector3& p_force, const Vector3& p_position) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply force to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (custom_integrator || p_force == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddForce(to_jolt(p_force), body->GetPosition() + to_jolt(p_position));

	_motion_changed();
}

void JoltBodyImpl3D::apply_central_force(const Vector3& p_force) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply central force to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (custom_integrator || p_force == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddForce(to_jolt(p_force));

	_motion_changed();
}

void JoltBodyImpl3D::apply_impulse(const Vector3& p_impulse, const Vector3& p_position) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply impulse to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (p_impulse == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddImpulse(to_jolt(p_impulse), body->GetPosition() + to_jolt(p_position));

	_motion_changed();
}

void JoltBodyImpl3D::apply_central_impulse(const Vector3& p_impulse) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply central impulse to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (p_impulse == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddImpulse(to_jolt(p_impulse));

	_motion_changed();
}

void JoltBodyImpl3D::apply_torque(const Vector3& p_torque) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply torque to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (custom_integrator || p_torque == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddTorque(to_jolt(p_torque));

	_motion_changed();
}

void JoltBodyImpl3D::apply_torque_impulse(const Vector3& p_impulse) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to apply torque impulse to '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	QUIET_FAIL_COND(!is_rigid());

	if (p_impulse == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->AddAngularImpulse(to_jolt(p_impulse));

	_motion_changed();
}

void JoltBodyImpl3D::add_constant_central_force(const Vector3& p_force) {
	if (p_force == Vector3()) {
		return;
	}

	constant_force += p_force;

	_motion_changed();
}

void JoltBodyImpl3D::add_constant_force(const Vector3& p_force, const Vector3& p_position) {
	if (p_force == Vector3()) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	const Vector3 center_of_mass = get_center_of_mass();
	const Vector3 body_position = get_position();
	const Vector3 center_of_mass_relative = center_of_mass - body_position;

	constant_force += p_force;
	constant_torque += (p_position - center_of_mass_relative).cross(p_force);

	_motion_changed();
}

void JoltBodyImpl3D::add_constant_torque(const Vector3& p_torque) {
	if (p_torque == Vector3()) {
		return;
	}

	constant_torque += p_torque;

	_motion_changed();
}

Vector3 JoltBodyImpl3D::get_constant_force() const {
	return constant_force;
}

void JoltBodyImpl3D::set_constant_force(const Vector3& p_force) {
	if (constant_force == p_force) {
		return;
	}

	constant_force = p_force;

	_motion_changed();
}

Vector3 JoltBodyImpl3D::get_constant_torque() const {
	return constant_torque;
}

void JoltBodyImpl3D::set_constant_torque(const Vector3& p_torque) {
	if (constant_torque == p_torque) {
		return;
	}

	constant_torque = p_torque;

	_motion_changed();
}

void JoltBodyImpl3D::add_collision_exception(const RID& p_excepted_body) {
	exceptions.push_back(p_excepted_body);

	_exceptions_changed();
}

void JoltBodyImpl3D::remove_collision_exception(const RID& p_excepted_body) {
	exceptions.erase(p_excepted_body);

	_exceptions_changed();
}

bool JoltBodyImpl3D::has_collision_exception(const RID& p_excepted_body) const {
	return exceptions.find(p_excepted_body) >= 0;
}

TypedArray<RID> JoltBodyImpl3D::get_collision_exceptions() const {
	TypedArray<RID> result;
	result.resize(exceptions.size());

	for (int32_t i = 0; i < exceptions.size(); ++i) {
		result[i] = exceptions[i];
	}

	return result;
}

void JoltBodyImpl3D::add_area(JoltAreaImpl3D* p_area) {
	areas.ordered_insert(p_area, [](const JoltAreaImpl3D* p_lhs, const JoltAreaImpl3D* p_rhs) {
		return p_lhs->get_priority() > p_rhs->get_priority();
	});

	_areas_changed();
}

void JoltBodyImpl3D::remove_area(JoltAreaImpl3D* p_area) {
	areas.erase(p_area);

	_areas_changed();
}

void JoltBodyImpl3D::add_joint(JoltJointImpl3D* p_joint) {
	joints.push_back(p_joint);

	_joints_changed();
}

void JoltBodyImpl3D::remove_joint(JoltJointImpl3D* p_joint) {
	joints.erase(p_joint);

	_joints_changed();
}

void JoltBodyImpl3D::call_queries([[maybe_unused]] JPH::Body& p_jolt_body) {
	if (!sync_state) {
		return;
	}

	if (custom_integration_callback.is_valid()) {
		if (custom_integration_userdata.get_type() != Variant::NIL) {
			static thread_local Array arguments = []() {
				Array array;
				array.resize(2);
				return array;
			}();

			arguments[0] = get_direct_state();
			arguments[1] = custom_integration_userdata;

			custom_integration_callback.callv(arguments);
		} else {
			static thread_local Array arguments = []() {
				Array array;
				array.resize(1);
				return array;
			}();

			arguments[0] = get_direct_state();

			custom_integration_callback.callv(arguments);
		}
	}

	if (body_state_callback.is_valid()) {
		static thread_local Array arguments = []() {
			Array array;
			array.resize(1);
			return array;
		}();

		arguments[0] = get_direct_state();

		body_state_callback.callv(arguments);
	}

	sync_state = false;
}

void JoltBodyImpl3D::pre_step(float p_step, JPH::Body& p_jolt_body) {
	JoltObjectImpl3D::pre_step(p_step, p_jolt_body);

	switch (mode) {
		case PhysicsServer3D::BODY_MODE_STATIC: {
			_pre_step_static(p_step, p_jolt_body);
		} break;
		case PhysicsServer3D::BODY_MODE_RIGID:
		case PhysicsServer3D::BODY_MODE_RIGID_LINEAR: {
			_pre_step_rigid(p_step, p_jolt_body);
		} break;
		case PhysicsServer3D::BODY_MODE_KINEMATIC: {
			_pre_step_kinematic(p_step, p_jolt_body);
		} break;
	}

	contact_count = 0;
}

void JoltBodyImpl3D::move_kinematic(float p_step, JPH::Body& p_jolt_body) {
	p_jolt_body.SetLinearVelocity(JPH::Vec3::sZero());
	p_jolt_body.SetAngularVelocity(JPH::Vec3::sZero());

	const JPH::RVec3 current_position = p_jolt_body.GetPosition();
	const JPH::Quat current_rotation = p_jolt_body.GetRotation();

	const JPH::RVec3 new_position = to_jolt_r(kinematic_transform.origin);
	const JPH::Quat new_rotation = to_jolt(kinematic_transform.basis);

	if (new_position == current_position && new_rotation == current_rotation) {
		return;
	}

	p_jolt_body.MoveKinematic(new_position, new_rotation, p_step);

	sync_state = true;
}

JoltPhysicsDirectBodyState3D* JoltBodyImpl3D::get_direct_state() {
	if (direct_state == nullptr) {
		direct_state = memnew(JoltPhysicsDirectBodyState3D(this));
	}

	return direct_state;
}

void JoltBodyImpl3D::set_mode(PhysicsServer3D::BodyMode p_mode) {
	if (p_mode == mode) {
		return;
	}

	ON_SCOPE_EXIT {
		_mode_changed();
	};

	mode = p_mode;

	if (space == nullptr) {
		return;
	}

	const JPH::EMotionType motion_type = _get_motion_type();

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	if (motion_type == JPH::EMotionType::Static) {
		put_to_sleep();
	}

	body->SetMotionType(motion_type);

	if (motion_type != JPH::EMotionType::Static) {
		wake_up();
	}

	if (motion_type == JPH::EMotionType::Kinematic) {
		body->SetLinearVelocity(JPH::Vec3::sZero());
		body->SetAngularVelocity(JPH::Vec3::sZero());
	}

	linear_surface_velocity = Vector3();
	angular_surface_velocity = Vector3();
}

bool JoltBodyImpl3D::is_ccd_enabled() const {
	if (space == nullptr) {
		return jolt_settings->mMotionQuality == JPH::EMotionQuality::LinearCast;
	}

	const JPH::BodyInterface& body_iface = space->get_body_iface();

	return body_iface.GetMotionQuality(jolt_id) == JPH::EMotionQuality::LinearCast;
}

void JoltBodyImpl3D::set_ccd_enabled(bool p_enabled) {
	const JPH::EMotionQuality motion_quality = p_enabled
		? JPH::EMotionQuality::LinearCast
		: JPH::EMotionQuality::Discrete;

	if (space == nullptr) {
		jolt_settings->mMotionQuality = motion_quality;
		return;
	}

	JPH::BodyInterface& body_iface = space->get_body_iface();

	body_iface.SetMotionQuality(jolt_id, motion_quality);
}

void JoltBodyImpl3D::set_mass(float p_mass) {
	if (p_mass != mass) {
		mass = p_mass;
		_update_mass_properties();
	}
}

void JoltBodyImpl3D::set_inertia(const Vector3& p_inertia) {
	if (p_inertia != inertia) {
		inertia = p_inertia;
		_update_mass_properties();
	}
}

float JoltBodyImpl3D::get_bounce() const {
	if (space == nullptr) {
		return jolt_settings->mRestitution;
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return body->GetRestitution();
}

void JoltBodyImpl3D::set_bounce(float p_bounce) {
	if (space == nullptr) {
		jolt_settings->mRestitution = p_bounce;
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->SetRestitution(p_bounce);
}

float JoltBodyImpl3D::get_friction() const {
	if (space == nullptr) {
		return jolt_settings->mFriction;
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return body->GetFriction();
}

void JoltBodyImpl3D::set_friction(float p_friction) {
	if (space == nullptr) {
		jolt_settings->mFriction = p_friction;
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->SetFriction(p_friction);
}

void JoltBodyImpl3D::set_gravity_scale(float p_scale) {
	if (gravity_scale == p_scale) {
		return;
	}

	gravity_scale = p_scale;

	_motion_changed();
}

void JoltBodyImpl3D::set_linear_damp(float p_damp) {
	if (p_damp < 0.0f) {
		WARN_PRINT(vformat(
			"Invalid linear damp for '%s'. "
			"Linear damp values less than 0 are not supported by Godot Jolt. "
			"Values outside this range will be clamped.",
			to_string()
		));

		p_damp = 0;
	}

	if (p_damp == linear_damp) {
		return;
	}

	linear_damp = p_damp;

	_update_damp();
}

void JoltBodyImpl3D::set_angular_damp(float p_damp) {
	if (p_damp < 0.0f) {
		WARN_PRINT(vformat(
			"Invalid angular damp for '%s'. "
			"Angular damp values less than 0 are not supported by Godot Jolt. "
			"Values outside this range will be clamped.",
			to_string()
		));

		p_damp = 0;
	}

	if (p_damp == angular_damp) {
		return;
	}

	angular_damp = p_damp;

	_update_damp();
}

bool JoltBodyImpl3D::is_axis_locked(PhysicsServer3D::BodyAxis p_axis) const {
	return (locked_axes & (uint32_t)p_axis) != 0;
}

void JoltBodyImpl3D::set_axis_lock(PhysicsServer3D::BodyAxis p_axis, bool p_enabled) {
	const uint32_t previous_locked_axes = locked_axes;

	if (p_enabled) {
		locked_axes |= (uint32_t)p_axis;
	} else {
		locked_axes &= ~(uint32_t)p_axis;
	}

	if (previous_locked_axes != locked_axes) {
		_axis_lock_changed();
	}
}

bool JoltBodyImpl3D::can_interact_with(const JoltBodyImpl3D& p_other) const {
	return (can_collide_with(p_other) || p_other.can_collide_with(*this)) &&
		!has_collision_exception(p_other.get_rid()) && !p_other.has_collision_exception(rid);
}

bool JoltBodyImpl3D::can_interact_with(const JoltSoftBodyImpl3D& p_other) const {
	return p_other.can_interact_with(*this);
}

bool JoltBodyImpl3D::can_interact_with(const JoltAreaImpl3D& p_other) const {
	return p_other.can_interact_with(*this);
}

JPH::BroadPhaseLayer JoltBodyImpl3D::_get_broad_phase_layer() const {
	switch (mode) {
		case PhysicsServer3D::BODY_MODE_STATIC: {
			return JoltBroadPhaseLayer::BODY_STATIC;
		}
		case PhysicsServer3D::BODY_MODE_KINEMATIC:
		case PhysicsServer3D::BODY_MODE_RIGID:
		case PhysicsServer3D::BODY_MODE_RIGID_LINEAR: {
			return JoltBroadPhaseLayer::BODY_DYNAMIC;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled body mode: '%d'", mode));
		}
	}
}

JPH::ObjectLayer JoltBodyImpl3D::_get_object_layer() const {
	ERR_FAIL_NULL_D(space);

	return space->map_to_object_layer(_get_broad_phase_layer(), collision_layer, collision_mask);
}

JPH::EMotionType JoltBodyImpl3D::_get_motion_type() const {
	switch (mode) {
		case PhysicsServer3D::BODY_MODE_STATIC: {
			return JPH::EMotionType::Static;
		}
		case PhysicsServer3D::BODY_MODE_KINEMATIC: {
			return JPH::EMotionType::Kinematic;
		}
		case PhysicsServer3D::BODY_MODE_RIGID:
		case PhysicsServer3D::BODY_MODE_RIGID_LINEAR: {
			return JPH::EMotionType::Dynamic;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled body mode: '%d'", mode));
		}
	}
}

void JoltBodyImpl3D::_add_to_space() {
	ON_SCOPE_EXIT {
		delete_safely(jolt_settings);
	};

	jolt_shape = build_shape();

	JPH::CollisionGroup::GroupID group_id = 0;
	JPH::CollisionGroup::SubGroupID sub_group_id = 0;
	JoltGroupFilter::encode_object(this, group_id, sub_group_id);

	jolt_settings->mUserData = reinterpret_cast<JPH::uint64>(this);
	jolt_settings->mObjectLayer = _get_object_layer();
	jolt_settings->mCollisionGroup = JPH::CollisionGroup(nullptr, group_id, sub_group_id);
	jolt_settings->mMotionType = _get_motion_type();
	jolt_settings->mAllowedDOFs = _calculate_allowed_dofs();
	jolt_settings->mAllowDynamicOrKinematic = true;
	jolt_settings->mCollideKinematicVsNonDynamic = reports_all_kinematic_contacts();
	jolt_settings->mUseManifoldReduction = !reports_contacts();
	jolt_settings->mLinearDamping = 0.0f;
	jolt_settings->mAngularDamping = 0.0f;
	jolt_settings->mMaxLinearVelocity = JoltProjectSettings::get_max_linear_velocity();
	jolt_settings->mMaxAngularVelocity = JoltProjectSettings::get_max_angular_velocity();

	if (JoltProjectSettings::use_enhanced_edge_removal()) {
		jolt_settings->mEnhancedInternalEdgeRemoval = true;
	}

	// HACK(mihe): We need to defer the setting of mass properties, to allow for modifying the
	// inverse inertia for the axis-locking, which we can't do until the body is created, so we set
	// it to some random values and calculate it properly once the body is created instead.
	jolt_settings->mOverrideMassProperties = JPH::EOverrideMassProperties::MassAndInertiaProvided;
	jolt_settings->mMassPropertiesOverride.mMass = 1.0f;
	jolt_settings->mMassPropertiesOverride.mInertia = JPH::Mat44::sIdentity();

	jolt_settings->SetShape(jolt_shape);

	const JPH::BodyID new_jolt_id = space->add_rigid_body(*this, *jolt_settings);
	QUIET_FAIL_COND(new_jolt_id.IsInvalid());

	jolt_id = new_jolt_id;
}

void JoltBodyImpl3D::_integrate_forces(float p_step, JPH::Body& p_jolt_body) {
	if (!p_jolt_body.IsActive()) {
		return;
	}

	_update_gravity(p_jolt_body);

	if (!custom_integrator) {
		JPH::MotionProperties& motion_properties = *p_jolt_body.GetMotionPropertiesUnchecked();

		JPH::Vec3 linear_velocity = motion_properties.GetLinearVelocity();
		JPH::Vec3 angular_velocity = motion_properties.GetAngularVelocity();

		// HACK(mihe): Jolt applies damping differently from Godot Physics, where Godot Physics
		// applies damping before integrating forces whereas Jolt does it after integrating forces.
		// The way Godot Physics does it seems to yield more consistent results across different
		// update frequencies when using high (>1) damping values, so we apply the damping ourselves
		// instead, before any force integration happens.

		linear_velocity *= MAX(1.0f - total_linear_damp * p_step, 0.0f);
		angular_velocity *= MAX(1.0f - total_angular_damp * p_step, 0.0f);

		linear_velocity += to_jolt(gravity) * p_step;

		motion_properties.SetLinearVelocityClamped(linear_velocity);
		motion_properties.SetAngularVelocityClamped(angular_velocity);

		p_jolt_body.AddForce(to_jolt(constant_force));
		p_jolt_body.AddTorque(to_jolt(constant_torque));
	}

	sync_state = true;
}

void JoltBodyImpl3D::_pre_step_static(
	[[maybe_unused]] float p_step,
	[[maybe_unused]] JPH::Body& p_jolt_body
) {
	// Nothing to do
}

void JoltBodyImpl3D::_pre_step_rigid(float p_step, JPH::Body& p_jolt_body) {
	_integrate_forces(p_step, p_jolt_body);
}

void JoltBodyImpl3D::_pre_step_kinematic(float p_step, JPH::Body& p_jolt_body) {
	_update_gravity(p_jolt_body);

	move_kinematic(p_step, p_jolt_body);

	if (reports_contacts()) {
		// HACK(mihe): This seems to emulate the behavior of Godot Physics, where kinematic bodies
		// are set as active (and thereby have their state synchronized on every step) only if its
		// max reported contacts is non-zero.
		sync_state = true;
	}
}

JPH::EAllowedDOFs JoltBodyImpl3D::_calculate_allowed_dofs() const {
	if (is_static()) {
		return JPH::EAllowedDOFs::All;
	}

	JPH::EAllowedDOFs allowed_dofs = JPH::EAllowedDOFs::All;

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_LINEAR_X)) {
		allowed_dofs &= ~JPH::EAllowedDOFs::TranslationX;
	}

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_LINEAR_Y)) {
		allowed_dofs &= ~JPH::EAllowedDOFs::TranslationY;
	}

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_LINEAR_Z)) {
		allowed_dofs &= ~JPH::EAllowedDOFs::TranslationZ;
	}

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_ANGULAR_X) || is_rigid_linear()) {
		allowed_dofs &= ~JPH::EAllowedDOFs::RotationX;
	}

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_ANGULAR_Y) || is_rigid_linear()) {
		allowed_dofs &= ~JPH::EAllowedDOFs::RotationY;
	}

	if (is_axis_locked(PhysicsServer3D::BODY_AXIS_ANGULAR_Z) || is_rigid_linear()) {
		allowed_dofs &= ~JPH::EAllowedDOFs::RotationZ;
	}

	ERR_FAIL_COND_V_MSG(
		allowed_dofs == JPH::EAllowedDOFs::None,
		JPH::EAllowedDOFs::All,
		vformat(
			"Invalid axis locks for '%s'. "
			"Locking all axes is not supported by Godot Jolt. "
			"All axes will be unlocked. "
			"Considering freezing the body as static instead.",
			to_string()
		)
	);

	return allowed_dofs;
}

JPH::MassProperties JoltBodyImpl3D::_calculate_mass_properties(const JPH::Shape& p_shape) const {
	const bool calculate_mass = mass <= 0;
	const bool calculate_inertia = inertia.x <= 0 || inertia.y <= 0 || inertia.z <= 0;

	JPH::MassProperties mass_properties = p_shape.GetMassProperties();

	if (calculate_mass && calculate_inertia) {
		// Use the mass properties calculated by the shape
	} else if (calculate_inertia) {
		mass_properties.ScaleToMass(mass);
	} else {
		mass_properties.mMass = mass;
	}

	if (inertia.x > 0) {
		mass_properties.mInertia(0, 0) = (float)inertia.x;
	}

	if (inertia.y > 0) {
		mass_properties.mInertia(1, 1) = (float)inertia.y;
	}

	if (inertia.z > 0) {
		mass_properties.mInertia(2, 2) = (float)inertia.z;
	}

	mass_properties.mInertia(3, 3) = 1.0f;

	return mass_properties;
}

JPH::MassProperties JoltBodyImpl3D::_calculate_mass_properties() const {
	return _calculate_mass_properties(*jolt_shape);
}

void JoltBodyImpl3D::_update_mass_properties() {
	if (space == nullptr) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetMotionPropertiesUnchecked()->SetMassProperties(
		_calculate_allowed_dofs(),
		_calculate_mass_properties()
	);
}

void JoltBodyImpl3D::_update_gravity(JPH::Body& p_jolt_body) {
	gravity = Vector3();

	const Vector3 position = to_godot(p_jolt_body.GetPosition());

	bool gravity_done = false;

	for (const JoltAreaImpl3D* area : areas) {
		gravity_done = integrate(gravity, area->get_gravity_mode(), [&]() {
			return area->compute_gravity(position);
		});

		if (gravity_done) {
			break;
		}
	}

	if (!gravity_done) {
		gravity += space->get_default_area()->compute_gravity(position);
	}

	gravity *= gravity_scale;
}

void JoltBodyImpl3D::_update_damp() {
	if (space == nullptr) {
		return;
	}

	total_linear_damp = 0.0;
	total_angular_damp = 0.0;

	bool linear_damp_done = linear_damp_mode == PhysicsServer3D::BODY_DAMP_MODE_REPLACE;
	bool angular_damp_done = angular_damp_mode == PhysicsServer3D::BODY_DAMP_MODE_REPLACE;

	for (const JoltAreaImpl3D* area : areas) {
		if (!linear_damp_done) {
			linear_damp_done = integrate(total_linear_damp, area->get_linear_damp_mode(), [&]() {
				return area->get_linear_damp();
			});
		}

		if (!angular_damp_done) {
			angular_damp_done = integrate(total_angular_damp, area->get_angular_damp_mode(), [&]() {
				return area->get_angular_damp();
			});
		}

		if (linear_damp_done && angular_damp_done) {
			break;
		}
	}

	const JoltAreaImpl3D* default_area = space->get_default_area();

	if (!linear_damp_done) {
		total_linear_damp += default_area->get_linear_damp();
	}

	if (!angular_damp_done) {
		total_angular_damp += default_area->get_angular_damp();
	}

	switch (linear_damp_mode) {
		case PhysicsServer3D::BODY_DAMP_MODE_COMBINE: {
			total_linear_damp += linear_damp;
		} break;
		case PhysicsServer3D::BODY_DAMP_MODE_REPLACE: {
			total_linear_damp = linear_damp;
		} break;
	}

	switch (angular_damp_mode) {
		case PhysicsServer3D::BODY_DAMP_MODE_COMBINE: {
			total_angular_damp += angular_damp;
		} break;
		case PhysicsServer3D::BODY_DAMP_MODE_REPLACE: {
			total_angular_damp = angular_damp;
		} break;
	}

	_motion_changed();
}

void JoltBodyImpl3D::_update_kinematic_transform() {
	if (is_kinematic()) {
		kinematic_transform = get_transform_unscaled();
	}
}

void JoltBodyImpl3D::_update_joint_constraints() {
	for (JoltJointImpl3D* joint : joints) {
		joint->rebuild();
	}
}

void JoltBodyImpl3D::_update_possible_kinematic_contacts() {
	const bool value = reports_all_kinematic_contacts();

	if (space == nullptr) {
		jolt_settings->mCollideKinematicVsNonDynamic = value;
	} else {
		const JoltWritableBody3D body = space->write_body(jolt_id);
		ERR_FAIL_COND(body.is_invalid());

		body->SetCollideKinematicVsNonDynamic(value);
	}
}

void JoltBodyImpl3D::_destroy_joint_constraints() {
	for (JoltJointImpl3D* joint : joints) {
		joint->destroy();
	}
}

void JoltBodyImpl3D::_update_group_filter() {
	JPH::GroupFilter* group_filter = !exceptions.is_empty() ? JoltGroupFilter::instance : nullptr;

	if (space == nullptr) {
		jolt_settings->mCollisionGroup.SetGroupFilter(group_filter);
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetCollisionGroup().SetGroupFilter(group_filter);
}

void JoltBodyImpl3D::_mode_changed() {
	_update_object_layer();
	_update_kinematic_transform();
	_update_mass_properties();
	wake_up();
}

void JoltBodyImpl3D::_shapes_built() {
	JoltShapedObjectImpl3D::_shapes_built();

	_update_mass_properties();
	_update_joint_constraints();
	wake_up();
}

void JoltBodyImpl3D::_space_changing() {
	JoltShapedObjectImpl3D::_space_changing();

	_destroy_joint_constraints();
}

void JoltBodyImpl3D::_space_changed() {
	JoltShapedObjectImpl3D::_space_changed();

	_update_kinematic_transform();
	_update_mass_properties();
	_update_group_filter();
	_update_joint_constraints();
	_areas_changed();

	sync_state = false;
}

void JoltBodyImpl3D::_areas_changed() {
	_update_damp();
	wake_up();
}

void JoltBodyImpl3D::_joints_changed() {
	wake_up();
}

void JoltBodyImpl3D::_transform_changed() {
	wake_up();
}

void JoltBodyImpl3D::_motion_changed() {
	wake_up();
}

void JoltBodyImpl3D::_exceptions_changed() {
	_update_group_filter();
}

void JoltBodyImpl3D::_axis_lock_changed() {
	_update_mass_properties();
	wake_up();
}

void JoltBodyImpl3D::_contact_reporting_changed() {
	_update_possible_kinematic_contacts();
	wake_up();
}
