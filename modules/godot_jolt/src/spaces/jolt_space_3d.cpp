#include "jolt_space_3d.hpp"

#include "joints/jolt_joint_impl_3d.hpp"
#include "objects/jolt_area_impl_3d.hpp"
#include "objects/jolt_body_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"
#include "servers/jolt_project_settings.hpp"
#include "shapes/jolt_custom_shape_type.hpp"
#include "shapes/jolt_shape_impl_3d.hpp"
#include "spaces/jolt_contact_listener_3d.hpp"
#include "spaces/jolt_layer_mapper.hpp"
#include "spaces/jolt_physics_direct_space_state_3d.hpp"
#include "spaces/jolt_temp_allocator.hpp"

namespace {

constexpr double DEFAULT_CONTACT_RECYCLE_RADIUS = 0.01;
constexpr double DEFAULT_CONTACT_MAX_SEPARATION = 0.05;
constexpr double DEFAULT_CONTACT_MAX_ALLOWED_PENETRATION = 0.01;
constexpr double DEFAULT_CONTACT_DEFAULT_BIAS = 0.8;
constexpr double DEFAULT_SLEEP_THRESHOLD_LINEAR = 0.1;
constexpr double DEFAULT_SLEEP_THRESHOLD_ANGULAR = 8.0 * Math_PI / 180;
constexpr double DEFAULT_SOLVER_ITERATIONS = 8;

} // namespace

JoltSpace3D::JoltSpace3D(JPH::JobSystem* p_job_system)
	: body_accessor(this)
	, job_system(p_job_system)
	, temp_allocator(new JoltTempAllocator())
	, layer_mapper(new JoltLayerMapper())
	, contact_listener(new JoltContactListener3D(this))
	, physics_system(new JPH::PhysicsSystem()) {
	physics_system->Init(
		(JPH::uint)JoltProjectSettings::get_max_bodies(),
		0,
		(JPH::uint)JoltProjectSettings::get_max_pairs(),
		(JPH::uint)JoltProjectSettings::get_max_contact_constraints(),
		*layer_mapper,
		*layer_mapper,
		*layer_mapper
	);

	JPH::PhysicsSettings settings;
	settings.mBaumgarte = JoltProjectSettings::get_position_correction();
	settings.mSpeculativeContactDistance = JoltProjectSettings::get_contact_distance();
	settings.mPenetrationSlop = JoltProjectSettings::get_contact_penetration();
	settings.mLinearCastThreshold = JoltProjectSettings::get_ccd_movement_threshold();
	settings.mLinearCastMaxPenetration = JoltProjectSettings::get_ccd_max_penetration();
	settings.mBodyPairCacheMaxDeltaPositionSq = JoltProjectSettings::get_pair_cache_distance();
	settings.mBodyPairCacheCosMaxDeltaRotationDiv2 = JoltProjectSettings::get_pair_cache_angle();
	settings.mNumVelocitySteps = (JPH::uint)JoltProjectSettings::get_velocity_iterations();
	settings.mNumPositionSteps = (JPH::uint)JoltProjectSettings::get_position_iterations();
	settings.mMinVelocityForRestitution = JoltProjectSettings::get_bounce_velocity_threshold();
	settings.mTimeBeforeSleep = JoltProjectSettings::get_sleep_time_threshold();
	settings.mPointVelocitySleepThreshold = JoltProjectSettings::get_sleep_velocity_threshold();
	settings.mUseBodyPairContactCache = JoltProjectSettings::is_pair_cache_enabled();
	settings.mAllowSleeping = JoltProjectSettings::is_sleep_enabled();

	physics_system->SetPhysicsSettings(settings);
	physics_system->SetGravity(JPH::Vec3::sZero());
	physics_system->SetContactListener(contact_listener);
	physics_system->SetSoftBodyContactListener(contact_listener);

	physics_system->SetCombineFriction(
		[](const JPH::Body& p_body1,
		   [[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id1,
		   const JPH::Body& p_body2,
		   [[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id2) {
			return ABS(MIN(p_body1.GetFriction(), p_body2.GetFriction()));
		}
	);

	physics_system->SetCombineRestitution(
		[](const JPH::Body& p_body1,
		   [[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id1,
		   const JPH::Body& p_body2,
		   [[maybe_unused]] const JPH::SubShapeID& p_sub_shape_id2) {
			return CLAMP(p_body1.GetRestitution() + p_body2.GetRestitution(), 0.0f, 1.0f);
		}
	);

#ifdef TOOLS_ENABLED
	// HACK(mihe): The `EditorLog` class gets initialized fairly late in the application flow, so if
	// we do this any earlier the warning is only ever going to be emitted to stdout and not the
	// editor log, hence why this is here.
	if (JoltProjectSettings::should_run_on_separate_thread()) {
		WARN_PRINT_ONCE(
			"Running on a separate thread is not currently supported by Godot Jolt. "
			"Any such setting will be ignored."
		);
	}
#endif // TOOLS_ENABLED
}

JoltSpace3D::~JoltSpace3D() {
	memdelete_safely(direct_state);
	delete_safely(physics_system);
	delete_safely(contact_listener);
	delete_safely(layer_mapper);
	delete_safely(temp_allocator);
}

void JoltSpace3D::step(float p_step) {
	last_step = p_step;

	_pre_step(p_step);

	const JPH::EPhysicsUpdateError
		update_error = physics_system->Update(p_step, 1, temp_allocator, job_system);

	if ((update_error & JPH::EPhysicsUpdateError::ManifoldCacheFull) !=
		JPH::EPhysicsUpdateError::None)
	{
		WARN_PRINT_ONCE(vformat(
			"Jolt's manifold cache exceeded capacity and contacts were ignored. "
			"Consider increasing maximum number of contact constraints in project settings. "
			"Maximum number of contact constraints is currently set to %d.",
			JoltProjectSettings::get_max_contact_constraints()
		));
	}

	if ((update_error & JPH::EPhysicsUpdateError::BodyPairCacheFull) !=
		JPH::EPhysicsUpdateError::None)
	{
		WARN_PRINT_ONCE(vformat(
			"Jolt's body pair cache exceeded capacity and contacts were ignored. "
			"Consider increasing maximum number of body pairs in project settings. "
			"Maximum number of body pairs is currently set to %d.",
			JoltProjectSettings::get_max_pairs()
		));
	}

	if ((update_error & JPH::EPhysicsUpdateError::ContactConstraintsFull) !=
		JPH::EPhysicsUpdateError::None)
	{
		WARN_PRINT_ONCE(vformat(
			"Jolt's contact constraint buffer exceeded capacity and contacts were ignored. "
			"Consider increasing maximum number of contact constraints in project settings. "
			"Maximum number of contact constraints is currently set to %d.",
			JoltProjectSettings::get_max_contact_constraints()
		));
	}

	_post_step(p_step);

	has_stepped = true;
	bodies_added_since_optimizing = 0;
}

void JoltSpace3D::call_queries() {
	if (!has_stepped) {
		// HACK(mihe): We need to skip the first invocation of this method, because there will be
		// pending notifications that need to be flushed first, which can cause weird conflicts with
		// things like `_integrate_forces`. This happens to also emulate the behavior of Godot
		// Physics, where (active) collision objects must register to have `call_queries` invoked,
		// which they don't do until the physics step, which happens after this.
		return;
	}

	body_accessor.acquire_all();

	const int32_t body_count = body_accessor.get_count();

	for (int32_t i = 0; i < body_count; ++i) {
		if (JPH::Body* jolt_body = body_accessor.try_get(i)) {
			if (!jolt_body->IsSensor() && !jolt_body->IsSoftBody()) {
				auto* body = reinterpret_cast<JoltBodyImpl3D*>(jolt_body->GetUserData());

				body->call_queries(*jolt_body);
			}
		}
	}

	for (int32_t i = 0; i < body_count; ++i) {
		if (JPH::Body* jolt_body = body_accessor.try_get(i)) {
			if (jolt_body->IsSensor()) {
				auto* area = reinterpret_cast<JoltAreaImpl3D*>(jolt_body->GetUserData());

				area->call_queries(*jolt_body);
			}
		}
	}

	body_accessor.release();
}

double JoltSpace3D::get_param(PhysicsServer3D::SpaceParameter p_param) const {
	switch (p_param) {
		case PhysicsServer3D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: {
			return DEFAULT_CONTACT_RECYCLE_RADIUS;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_SEPARATION: {
			return DEFAULT_CONTACT_MAX_SEPARATION;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION: {
			return DEFAULT_CONTACT_MAX_ALLOWED_PENETRATION;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_DEFAULT_BIAS: {
			return DEFAULT_CONTACT_DEFAULT_BIAS;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: {
			return DEFAULT_SLEEP_THRESHOLD_LINEAR;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: {
			return DEFAULT_SLEEP_THRESHOLD_ANGULAR;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_TIME_TO_SLEEP: {
			return JoltProjectSettings::get_sleep_time_threshold();
		}
		case PhysicsServer3D::SPACE_PARAM_SOLVER_ITERATIONS: {
			return DEFAULT_SOLVER_ITERATIONS;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled space parameter: '%d'", p_param));
		}
	}
}

void JoltSpace3D::set_param(
	PhysicsServer3D::SpaceParameter p_param,
	[[maybe_unused]] double p_value
) {
	switch (p_param) {
		case PhysicsServer3D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: {
			WARN_PRINT(
				"Space-specific contact recycle radius is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_SEPARATION: {
			WARN_PRINT(
				"Space-specific contact max separation is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION: {
			WARN_PRINT(
				"Space-specific contact max allowed penetration is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_DEFAULT_BIAS: {
			WARN_PRINT(
				"Space-specific contact default bias is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: {
			WARN_PRINT(
				"Space-specific linear velocity sleep threshold is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: {
			WARN_PRINT(
				"Space-specific angular velocity sleep threshold is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_TIME_TO_SLEEP: {
			WARN_PRINT(
				"Space-specific body sleep time is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		case PhysicsServer3D::SPACE_PARAM_SOLVER_ITERATIONS: {
			WARN_PRINT(
				"Space-specific solver iterations is not supported by Godot Jolt. "
				"Any such value will be ignored."
			);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled space parameter: '%d'", p_param));
		} break;
	}
}

JPH::BodyInterface& JoltSpace3D::get_body_iface() {
	return physics_system->GetBodyInterfaceNoLock();
}

const JPH::BodyInterface& JoltSpace3D::get_body_iface() const {
	return physics_system->GetBodyInterfaceNoLock();
}

const JPH::BodyLockInterface& JoltSpace3D::get_lock_iface() const {
	return physics_system->GetBodyLockInterfaceNoLock();
}

const JPH::BroadPhaseQuery& JoltSpace3D::get_broad_phase_query() const {
	return physics_system->GetBroadPhaseQuery();
}

const JPH::NarrowPhaseQuery& JoltSpace3D::get_narrow_phase_query() const {
	return physics_system->GetNarrowPhaseQueryNoLock();
}

JPH::ObjectLayer JoltSpace3D::map_to_object_layer(
	JPH::BroadPhaseLayer p_broad_phase_layer,
	uint32_t p_collision_layer,
	uint32_t p_collision_mask
) {
	return layer_mapper->to_object_layer(p_broad_phase_layer, p_collision_layer, p_collision_mask);
}

void JoltSpace3D::map_from_object_layer(
	JPH::ObjectLayer p_object_layer,
	JPH::BroadPhaseLayer& p_broad_phase_layer,
	uint32_t& p_collision_layer,
	uint32_t& p_collision_mask
) const {
	layer_mapper->from_object_layer(
		p_object_layer,
		p_broad_phase_layer,
		p_collision_layer,
		p_collision_mask
	);
}

JoltReadableBody3D JoltSpace3D::read_body(const JPH::BodyID& p_body_id) const {
	return {*this, p_body_id};
}

JoltReadableBody3D JoltSpace3D::read_body(const JoltObjectImpl3D& p_object) const {
	return read_body(p_object.get_jolt_id());
}

JoltWritableBody3D JoltSpace3D::write_body(const JPH::BodyID& p_body_id) const {
	return {*this, p_body_id};
}

JoltWritableBody3D JoltSpace3D::write_body(const JoltObjectImpl3D& p_object) const {
	return write_body(p_object.get_jolt_id());
}

JoltReadableBodies3D JoltSpace3D::read_bodies(const JPH::BodyID* p_body_ids, int32_t p_body_count)
	const {
	return {*this, p_body_ids, p_body_count};
}

JoltWritableBodies3D JoltSpace3D::write_bodies(const JPH::BodyID* p_body_ids, int32_t p_body_count)
	const {
	return {*this, p_body_ids, p_body_count};
}

JoltPhysicsDirectSpaceState3D* JoltSpace3D::get_direct_state() {
	if (direct_state == nullptr) {
		direct_state = memnew(JoltPhysicsDirectSpaceState3D(this));
	}

	return direct_state;
}

void JoltSpace3D::set_default_area(JoltAreaImpl3D* p_area) {
	if (default_area == p_area) {
		return;
	}

	if (default_area != nullptr) {
		default_area->set_default_area(false);
	}

	default_area = p_area;

	if (default_area != nullptr) {
		default_area->set_default_area(true);
	}
}

JPH::BodyID JoltSpace3D::add_rigid_body(
	const JoltObjectImpl3D& p_object,
	const JPH::BodyCreationSettings& p_settings
) {
	const JPH::BodyID body_id = get_body_iface().CreateAndAddBody(
		p_settings,
		// HACK(mihe): Since `BODY_STATE_TRANSFORM` will be set right after creation it's more or
		// less impossible to have a body be sleeping when created, so we default to always starting
		// out as awake/active.
		JPH::EActivation::Activate
	);

	ERR_FAIL_COND_D_MSG(
		body_id.IsInvalid(),
		vformat(
			"Failed to create underlying Jolt body for '%s'. "
			"Consider increasing maximum number of bodies in project settings. "
			"Maximum number of bodies is currently set to %d.",
			p_object.to_string(),
			JoltProjectSettings::get_max_bodies()
		)
	);

	bodies_added_since_optimizing += 1;

	return body_id;
}

JPH::BodyID JoltSpace3D::add_soft_body(
	const JoltObjectImpl3D& p_object,
	const JPH::SoftBodyCreationSettings& p_settings
) {
	const JPH::BodyID body_id = get_body_iface().CreateAndAddSoftBody(
		p_settings,
		JPH::EActivation::Activate
	);

	ERR_FAIL_COND_D_MSG(
		body_id.IsInvalid(),
		vformat(
			"Failed to create underlying Jolt body for '%s'. "
			"Consider increasing maximum number of bodies in project settings. "
			"Maximum number of bodies is currently set to %d.",
			p_object.to_string(),
			JoltProjectSettings::get_max_bodies()
		)
	);

	bodies_added_since_optimizing += 1;

	return body_id;
}

void JoltSpace3D::remove_body(const JPH::BodyID& p_body_id) {
	JPH::BodyInterface& body_iface = get_body_iface();

	body_iface.RemoveBody(p_body_id);
	body_iface.DestroyBody(p_body_id);
}

void JoltSpace3D::try_optimize() {
	// HACK(mihe): This makes assumptions about the underlying acceleration structure of Jolt's
	// broad-phase, which currently uses a quadtree, and which gets walked with a fixed-size node
	// stack of 128. This means that when the quadtree is completely unbalanced, as is the case if
	// we add bodies one-by-one without ever stepping the simulation, like in the editor viewport,
	// we would exceed this stack size (resulting in an incomplete search) as soon as we perform a
	// physics query after having added somewhere in the order of 128 * 3 bodies. We leave a hefty
	// margin just in case.

	if (likely(bodies_added_since_optimizing < 128)) {
		return;
	}

	physics_system->OptimizeBroadPhase();

	bodies_added_since_optimizing = 0;
}

void JoltSpace3D::add_joint(JPH::Constraint* p_jolt_ref) {
	physics_system->AddConstraint(p_jolt_ref);
}

void JoltSpace3D::add_joint(JoltJointImpl3D* p_joint) {
	add_joint(p_joint->get_jolt_ref());
}

void JoltSpace3D::remove_joint(JPH::Constraint* p_jolt_ref) {
	physics_system->RemoveConstraint(p_jolt_ref);
}

void JoltSpace3D::remove_joint(JoltJointImpl3D* p_joint) {
	remove_joint(p_joint->get_jolt_ref());
}

#ifdef TOOLS_ENABLED

void JoltSpace3D::dump_debug_snapshot(const String& p_dir) {
	const Dictionary datetime = Time::get_singleton()->get_datetime_dict_from_system();

	const String datetime_str = vformat(
		"%04d-%02d-%02d_%02d-%02d-%02d",
		datetime["year"],
		datetime["month"],
		datetime["day"],
		datetime["hour"],
		datetime["minute"],
		datetime["second"]
	);

	const String path = p_dir + vformat("/jolt_snapshot_%s_%d.bin", datetime_str, rid.get_id());

	Ref<FileAccess> file_access = FileAccess::open(path, FileAccess::ModeFlags::WRITE);

	ERR_FAIL_NULL_MSG(
		file_access,
		vformat(
			"Failed to open '%s' for writing when saving snapshot of physics space with RID '%d'.",
			path,
			rid.get_id()
		)
	);

	JPH::PhysicsScene physics_scene;
	physics_scene.FromPhysicsSystem(physics_system);

	for (JPH::BodyCreationSettings& settings : physics_scene.GetBodies()) {
		const auto* object = reinterpret_cast<const JoltObjectImpl3D*>(settings.mUserData);

		if (const JoltBodyImpl3D* body = object->as_body()) {
			// HACK(mihe): Since we do our own integration of gravity and damping, while leaving
			// Jolt's own values at zero, we need to transfer over the correct values.
			settings.mGravityFactor = body->get_gravity_scale();
			settings.mLinearDamping = body->get_total_linear_damp();
			settings.mAngularDamping = body->get_total_angular_damp();
		}

		settings.SetShape(JoltShapeImpl3D::without_custom_shapes(settings.GetShape()));
	}

	JoltStreamOutWrapper output_stream(file_access);
	physics_scene.SaveBinaryState(output_stream, true, false);

	ERR_FAIL_COND_MSG(
		file_access->get_error() != OK,
		vformat(
			"Writing snapshot of physics space with RID '%d' to '%s' failed with error '%s'.",
			rid.get_id(),
			path,
			VariantUtilityFunctions::error_string(file_access->get_error())
		)
	);

	print_line(
		vformat("Snapshot of physics space with RID '%d' saved to '%s'.", rid.get_id(), path)
	);
}

const PackedVector3Array& JoltSpace3D::get_debug_contacts() const {
	return contact_listener->get_debug_contacts();
}

int32_t JoltSpace3D::get_debug_contact_count() const {
	return contact_listener->get_debug_contact_count();
}

int32_t JoltSpace3D::get_max_debug_contacts() const {
	return contact_listener->get_max_debug_contacts();
}

void JoltSpace3D::set_max_debug_contacts(int32_t p_count) {
	contact_listener->set_max_debug_contacts(p_count);
}

#endif // TOOLS_ENABLED

void JoltSpace3D::_pre_step(float p_step) {
	body_accessor.acquire_all();

	contact_listener->pre_step();

	const int32_t body_count = body_accessor.get_count();

	for (int32_t i = 0; i < body_count; ++i) {
		if (JPH::Body* jolt_body = body_accessor.try_get(i)) {
			if (jolt_body->IsSoftBody()) {
				continue;
			}

			auto* object = reinterpret_cast<JoltShapedObjectImpl3D*>(jolt_body->GetUserData());

			object->pre_step(p_step, *jolt_body);

			if (object->reports_contacts()) {
				contact_listener->listen_for(object);
			}
		}
	}

	body_accessor.release();
}

void JoltSpace3D::_post_step(float p_step) {
	body_accessor.acquire_all();

	contact_listener->post_step();

	const int32_t body_count = body_accessor.get_count();

	for (int32_t i = 0; i < body_count; ++i) {
		if (JPH::Body* jolt_body = body_accessor.try_get(i)) {
			if (jolt_body->IsSoftBody()) {
				continue;
			}

			auto* object = reinterpret_cast<JoltObjectImpl3D*>(jolt_body->GetUserData());

			object->post_step(p_step, *jolt_body);
		}
	}

	body_accessor.release();
}
