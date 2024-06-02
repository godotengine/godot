#include "jolt_area_impl_3d.hpp"

#include "objects/jolt_body_impl_3d.hpp"
#include "objects/jolt_group_filter.hpp"
#include "objects/jolt_soft_body_impl_3d.hpp"
#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_broad_phase_layer.hpp"
#include "spaces/jolt_space_3d.hpp"

namespace {

constexpr double DEFAULT_WIND_FORCE_MAGNITUDE = 0.0;
constexpr double DEFAULT_WIND_ATTENUATION_FACTOR = 0.0;

const Vector3 DEFAULT_WIND_SOURCE = {};
const Vector3 DEFAULT_WIND_DIRECTION = {};

} // namespace

JoltAreaImpl3D::JoltAreaImpl3D()
	: JoltShapedObjectImpl3D(OBJECT_TYPE_AREA) { }

bool JoltAreaImpl3D::is_default_area() const {
	return space != nullptr && space->get_default_area() == this;
}

void JoltAreaImpl3D::set_default_area(bool p_value) {
	if (p_value) {
		_update_default_gravity();
	}
}

void JoltAreaImpl3D::set_transform(const Transform3D& p_transform) {
	Vector3 new_scale;
	const Transform3D new_transform = decomposed(p_transform, new_scale);

	if (!scale.is_equal_approx(new_scale)) {
		scale = new_scale;
		_shapes_changed();
	}

	if (space == nullptr) {
		jolt_settings->mPosition = to_jolt_r(new_transform.origin);
		jolt_settings->mRotation = to_jolt(new_transform.basis);
	} else {
		space->get_body_iface().SetPositionAndRotation(
			jolt_id,
			to_jolt_r(new_transform.origin),
			to_jolt(new_transform.basis),
			JPH::EActivation::DontActivate
		);
	}
}

Variant JoltAreaImpl3D::get_param(PhysicsServer3D::AreaParameter p_param) const {
	switch (p_param) {
		case PhysicsServer3D::AREA_PARAM_GRAVITY_OVERRIDE_MODE: {
			return get_gravity_mode();
		}
		case PhysicsServer3D::AREA_PARAM_GRAVITY: {
			return get_gravity();
		}
		case PhysicsServer3D::AREA_PARAM_GRAVITY_VECTOR: {
			return get_gravity_vector();
		}
		case PhysicsServer3D::AREA_PARAM_GRAVITY_IS_POINT: {
			return is_point_gravity();
		}
		case PhysicsServer3D::AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE: {
			return get_point_gravity_distance();
		}
		case PhysicsServer3D::AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE: {
			return get_linear_damp_mode();
		}
		case PhysicsServer3D::AREA_PARAM_LINEAR_DAMP: {
			return get_linear_damp();
		}
		case PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE: {
			return get_angular_damp_mode();
		}
		case PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP: {
			return get_angular_damp();
		}
		case PhysicsServer3D::AREA_PARAM_PRIORITY: {
			return get_priority();
		}
		case PhysicsServer3D::AREA_PARAM_WIND_FORCE_MAGNITUDE: {
			return DEFAULT_WIND_FORCE_MAGNITUDE;
		}
		case PhysicsServer3D::AREA_PARAM_WIND_SOURCE: {
			return DEFAULT_WIND_SOURCE;
		}
		case PhysicsServer3D::AREA_PARAM_WIND_DIRECTION: {
			return DEFAULT_WIND_DIRECTION;
		}
		case PhysicsServer3D::AREA_PARAM_WIND_ATTENUATION_FACTOR: {
			return DEFAULT_WIND_ATTENUATION_FACTOR;
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled area parameter: '%d'", p_param));
		}
	}
}

void JoltAreaImpl3D::set_param(PhysicsServer3D::AreaParameter p_param, const Variant& p_value) {
	switch (p_param) {
		case PhysicsServer3D::AREA_PARAM_GRAVITY_OVERRIDE_MODE: {
			set_gravity_mode((OverrideMode)(int32_t)p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_GRAVITY: {
			set_gravity(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_GRAVITY_VECTOR: {
			set_gravity_vector(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_GRAVITY_IS_POINT: {
			set_point_gravity(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_GRAVITY_POINT_UNIT_DISTANCE: {
			set_point_gravity_distance(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_LINEAR_DAMP_OVERRIDE_MODE: {
			set_linear_damp_mode((OverrideMode)(int32_t)p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_LINEAR_DAMP: {
			set_area_linear_damp(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP_OVERRIDE_MODE: {
			set_angular_damp_mode((OverrideMode)(int32_t)p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_ANGULAR_DAMP: {
			set_area_angular_damp(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_PRIORITY: {
			set_priority(p_value);
		} break;
		case PhysicsServer3D::AREA_PARAM_WIND_FORCE_MAGNITUDE: {
			if (!Math::is_equal_approx((double)p_value, DEFAULT_WIND_FORCE_MAGNITUDE)) {
				WARN_PRINT(vformat(
					"Invalid wind force magnitude for '%s'. "
					"Area wind force magnitude is not supported by Godot Jolt. "
					"Any such value will be ignored.",
					to_string()
				));
			}
		} break;
		case PhysicsServer3D::AREA_PARAM_WIND_SOURCE: {
			if (!((Vector3)p_value).is_equal_approx(DEFAULT_WIND_SOURCE)) {
				WARN_PRINT(vformat(
					"Invalid wind source for '%s'. "
					"Area wind source is not supported by Godot Jolt. "
					"Any such value will be ignored.",
					to_string()
				));
			}
		} break;
		case PhysicsServer3D::AREA_PARAM_WIND_DIRECTION: {
			if (!((Vector3)p_value).is_equal_approx(DEFAULT_WIND_DIRECTION)) {
				WARN_PRINT(vformat(
					"Invalid wind direction for '%s'. "
					"Area wind direction is not supported by Godot Jolt. "
					"Any such value will be ignored.",
					to_string()
				));
			}
		} break;
		case PhysicsServer3D::AREA_PARAM_WIND_ATTENUATION_FACTOR: {
			if (!Math::is_equal_approx((double)p_value, DEFAULT_WIND_ATTENUATION_FACTOR)) {
				WARN_PRINT(vformat(
					"Invalid wind attenuation for '%s'. "
					"Area wind attenuation is not supported by Godot Jolt. "
					"Any such value will be ignored.",
					to_string()
				));
			}
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled area parameter: '%d'", p_param));
		} break;
	}
}

void JoltAreaImpl3D::set_body_monitor_callback(const Callable& p_callback) {
	if (p_callback == body_monitor_callback) {
		return;
	}

	body_monitor_callback = p_callback;

	_body_monitoring_changed();
}

void JoltAreaImpl3D::set_area_monitor_callback(const Callable& p_callback) {
	if (p_callback == area_monitor_callback) {
		return;
	}

	area_monitor_callback = p_callback;

	_area_monitoring_changed();
}

void JoltAreaImpl3D::set_monitorable(bool p_monitorable) {
	if (p_monitorable == monitorable) {
		return;
	}

	monitorable = p_monitorable;

	_monitorable_changed();
}

bool JoltAreaImpl3D::can_monitor(const JoltBodyImpl3D& p_other) const {
	return (collision_mask & p_other.get_collision_layer()) != 0;
}

bool JoltAreaImpl3D::can_monitor([[maybe_unused]] const JoltSoftBodyImpl3D& p_other) const {
	return false;
}

bool JoltAreaImpl3D::can_monitor(const JoltAreaImpl3D& p_other) const {
	return p_other.is_monitorable() && (collision_mask & p_other.get_collision_layer()) != 0;
}

bool JoltAreaImpl3D::can_interact_with(const JoltBodyImpl3D& p_other) const {
	return can_monitor(p_other);
}

bool JoltAreaImpl3D::can_interact_with([[maybe_unused]] const JoltSoftBodyImpl3D& p_other) const {
	return false;
}

bool JoltAreaImpl3D::can_interact_with(const JoltAreaImpl3D& p_other) const {
	return can_monitor(p_other) || p_other.can_monitor(*this);
}

Vector3 JoltAreaImpl3D::get_velocity_at_position([[maybe_unused]] const Vector3& p_position) const {
	return {0.0f, 0.0f, 0.0f};
}

void JoltAreaImpl3D::set_point_gravity(bool p_enabled) {
	if (point_gravity == p_enabled) {
		return;
	}

	point_gravity = p_enabled;

	_gravity_changed();
}

void JoltAreaImpl3D::set_gravity(float p_gravity) {
	if (gravity == p_gravity) {
		return;
	}

	gravity = p_gravity;

	_gravity_changed();
}

void JoltAreaImpl3D::set_point_gravity_distance(float p_distance) {
	if (point_gravity_distance == p_distance) {
		return;
	}

	point_gravity_distance = p_distance;

	_gravity_changed();
}

void JoltAreaImpl3D::set_gravity_mode(OverrideMode p_mode) {
	if (gravity_mode == p_mode) {
		return;
	}

	gravity_mode = p_mode;

	_gravity_changed();
}

void JoltAreaImpl3D::set_gravity_vector(const Vector3& p_vector) {
	if (gravity_vector == p_vector) {
		return;
	}

	gravity_vector = p_vector;

	_gravity_changed();
}

Vector3 JoltAreaImpl3D::compute_gravity(const Vector3& p_position) const {
	if (!point_gravity) {
		return gravity_vector * gravity;
	}

	const Vector3 point = get_transform_scaled().xform(gravity_vector);
	const Vector3 to_point = point - p_position;
	const real_t to_point_dist_sq = MAX(to_point.length_squared(), (real_t)CMP_EPSILON);
	const Vector3 to_point_dir = to_point / Math::sqrt(to_point_dist_sq);

	if (point_gravity_distance == 0.0f) {
		return to_point_dir * gravity;
	}

	const float gravity_dist_sq = point_gravity_distance * point_gravity_distance;

	return to_point_dir * (gravity * gravity_dist_sq / to_point_dist_sq);
}

void JoltAreaImpl3D::body_shape_entered(
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	Overlap& overlap = bodies_by_id[p_body_id];

	if (overlap.shape_pairs.is_empty()) {
		_notify_body_entered(p_body_id);
	}

	_add_shape_pair(overlap, p_body_id, p_other_shape_id, p_self_shape_id);
}

bool JoltAreaImpl3D::body_shape_exited(
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	Overlap* overlap = bodies_by_id.getptr(p_body_id);

	if (overlap == nullptr) {
		return false;
	}

	if (!_remove_shape_pair(*overlap, p_other_shape_id, p_self_shape_id)) {
		return false;
	}

	if (overlap->shape_pairs.is_empty()) {
		_notify_body_exited(p_body_id);
	}

	return true;
}

void JoltAreaImpl3D::area_shape_entered(
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	_add_shape_pair(areas_by_id[p_body_id], p_body_id, p_other_shape_id, p_self_shape_id);
}

bool JoltAreaImpl3D::area_shape_exited(
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	Overlap* overlap = areas_by_id.getptr(p_body_id);

	if (overlap == nullptr) {
		return false;
	}

	return _remove_shape_pair(*overlap, p_other_shape_id, p_self_shape_id);
}

bool JoltAreaImpl3D::shape_exited(
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	return body_shape_exited(p_body_id, p_other_shape_id, p_self_shape_id) ||
		area_shape_exited(p_body_id, p_other_shape_id, p_self_shape_id);
}

void JoltAreaImpl3D::call_queries([[maybe_unused]] JPH::Body& p_jolt_body) {
	_flush_events(bodies_by_id, body_monitor_callback);
	_flush_events(areas_by_id, area_monitor_callback);
}

JPH::BroadPhaseLayer JoltAreaImpl3D::_get_broad_phase_layer() const {
	return monitorable
		? JoltBroadPhaseLayer::AREA_DETECTABLE
		: JoltBroadPhaseLayer::AREA_UNDETECTABLE;
}

JPH::ObjectLayer JoltAreaImpl3D::_get_object_layer() const {
	ERR_FAIL_NULL_D(space);

	return space->map_to_object_layer(_get_broad_phase_layer(), collision_layer, collision_mask);
}

void JoltAreaImpl3D::_add_to_space() {
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
	jolt_settings->mIsSensor = true;
	jolt_settings->mUseManifoldReduction = false;

	if (JoltProjectSettings::areas_detect_static_bodies()) {
		jolt_settings->mCollideKinematicVsNonDynamic = true;
	}

	jolt_settings->SetShape(build_shape());

	const JPH::BodyID new_jolt_id = space->add_rigid_body(*this, *jolt_settings);
	QUIET_FAIL_COND(new_jolt_id.IsInvalid());

	jolt_id = new_jolt_id;
}

void JoltAreaImpl3D::_add_shape_pair(
	Overlap& p_overlap,
	const JPH::BodyID& p_body_id,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	const JoltReadableBody3D other_jolt_body = space->read_body(p_body_id);
	const JoltShapedObjectImpl3D* other_object = other_jolt_body.as_shaped();
	ERR_FAIL_NULL(other_object);

	p_overlap.rid = other_object->get_rid();
	p_overlap.instance_id = other_object->get_instance_id();

	ShapeIndexPair& shape_indices = p_overlap.shape_pairs[{p_other_shape_id, p_self_shape_id}];

	shape_indices.other = other_object->find_shape_index(p_other_shape_id);
	shape_indices.self = find_shape_index(p_self_shape_id);

	p_overlap.pending_added.push_back(shape_indices);
}

bool JoltAreaImpl3D::_remove_shape_pair(
	Overlap& p_overlap,
	const JPH::SubShapeID& p_other_shape_id,
	const JPH::SubShapeID& p_self_shape_id
) {
	auto shape_pair = p_overlap.shape_pairs.find({p_other_shape_id, p_self_shape_id});

	if (shape_pair == p_overlap.shape_pairs.end()) {
		return false;
	}

	p_overlap.pending_removed.push_back(shape_pair->second);
	p_overlap.shape_pairs.remove(shape_pair);

	return true;
}

void JoltAreaImpl3D::_flush_events(OverlapsById& p_objects, const Callable& p_callback) {
	p_objects.erase_if([&](auto& p_pair) {
		auto& [id, overlap] = p_pair;

		if (p_callback.is_valid()) {
			for (auto& shape_indices : overlap.pending_removed) {
				_report_event(
					p_callback,
					PhysicsServer3D::AREA_BODY_REMOVED,
					overlap.rid,
					overlap.instance_id,
					shape_indices.other,
					shape_indices.self
				);
			}

			for (auto& shape_indices : overlap.pending_added) {
				_report_event(
					p_callback,
					PhysicsServer3D::AREA_BODY_ADDED,
					overlap.rid,
					overlap.instance_id,
					shape_indices.other,
					shape_indices.self
				);
			}
		}

		overlap.pending_removed.clear();
		overlap.pending_added.clear();

		return overlap.shape_pairs.is_empty();
	});
}

void JoltAreaImpl3D::_report_event(
	const Callable& p_callback,
	PhysicsServer3D::AreaBodyStatus p_status,
	const RID& p_other_rid,
	ObjectID p_other_instance_id,
	int32_t p_other_shape_index,
	int32_t p_self_shape_index
) const {
	ERR_FAIL_COND(!p_callback.is_valid());

	static thread_local Array arguments = []() {
		Array array;
		array.resize(5);
		return array;
	}();

	arguments[0] = p_status;
	arguments[1] = p_other_rid;
	arguments[2] = p_other_instance_id;
	arguments[3] = p_other_shape_index;
	arguments[4] = p_self_shape_index;

	p_callback.callv(arguments);
}

void JoltAreaImpl3D::_notify_body_entered(const JPH::BodyID& p_body_id) {
	const JoltReadableBody3D jolt_body = space->read_body(p_body_id);

	JoltBodyImpl3D* body = jolt_body.as_body();
	QUIET_FAIL_NULL(body);

	body->add_area(this);
}

void JoltAreaImpl3D::_notify_body_exited(const JPH::BodyID& p_body_id) {
	const JoltReadableBody3D jolt_body = space->read_body(p_body_id);

	JoltBodyImpl3D* body = jolt_body.as_body();
	QUIET_FAIL_NULL(body);

	body->remove_area(this);
}

void JoltAreaImpl3D::_force_bodies_entered() {
	for (auto& [id, body] : bodies_by_id) {
		for (const auto& [id_pair, index_pair] : body.shape_pairs) {
			body.pending_added.push_back(index_pair);
		}
	}
}

void JoltAreaImpl3D::_force_bodies_exited(bool p_remove) {
	for (auto& [id, body] : bodies_by_id) {
		for (const auto& [id_pair, index_pair] : body.shape_pairs) {
			body.pending_removed.push_back(index_pair);
		}

		if (p_remove) {
			body.shape_pairs.clear();
			_notify_body_exited(id);
		}
	}
}

void JoltAreaImpl3D::_force_areas_entered() {
	for (auto& [id, area] : areas_by_id) {
		for (const auto& [id_pair, index_pair] : area.shape_pairs) {
			area.pending_added.push_back(index_pair);
		}
	}
}

void JoltAreaImpl3D::_force_areas_exited(bool p_remove) {
	for (auto& [id, area] : areas_by_id) {
		for (const auto& [id_pair, index_pair] : area.shape_pairs) {
			area.pending_removed.push_back(index_pair);
		}

		if (p_remove) {
			area.shape_pairs.clear();
		}
	}
}

void JoltAreaImpl3D::_update_group_filter() {
	if (space == nullptr) {
		return;
	}

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetCollisionGroup().SetGroupFilter(JoltGroupFilter::instance);
}

void JoltAreaImpl3D::_update_default_gravity() {
	if (is_default_area()) {
		space->get_physics_system().SetGravity(to_jolt(gravity_vector) * gravity);
	}
}

void JoltAreaImpl3D::_space_changing() {
	JoltShapedObjectImpl3D::_space_changing();

	if (space != nullptr) {
		// HACK(mihe): Ideally we would rely on our contact listener to report all the exits when we
		// move between (or out of) spaces, but because our Jolt body is going to be destroyed when
		// we leave this space the contact listener won't be able to retrieve the corresponding area
		// and as such cannot report any exits, so we're forced to do it manually instead.
		_force_bodies_exited(true);
		_force_areas_exited(true);
	}
}

void JoltAreaImpl3D::_space_changed() {
	JoltShapedObjectImpl3D::_space_changed();

	_update_group_filter();
	_update_default_gravity();
}

void JoltAreaImpl3D::_body_monitoring_changed() {
	if (has_body_monitor_callback()) {
		_force_bodies_entered();
	} else {
		_force_bodies_exited(false);
	}
}

void JoltAreaImpl3D::_area_monitoring_changed() {
	if (has_area_monitor_callback()) {
		_force_areas_entered();
	} else {
		_force_areas_exited(false);
	}
}

void JoltAreaImpl3D::_monitorable_changed() {
	_update_object_layer();
}

void JoltAreaImpl3D::_gravity_changed() {
	_update_default_gravity();
}
