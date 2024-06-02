#include "jolt_soft_body_impl_3d.hpp"

#include "objects/jolt_area_impl_3d.hpp"
#include "objects/jolt_body_impl_3d.hpp"
#include "objects/jolt_group_filter.hpp"
#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_broad_phase_layer.hpp"
#include "spaces/jolt_space_3d.hpp"

JoltSoftBodyImpl3D::JoltSoftBodyImpl3D()
	: JoltObjectImpl3D(OBJECT_TYPE_SOFT_BODY) {
	jolt_settings->mRestitution = 0.0f;
	jolt_settings->mFriction = 1.0f;
	jolt_settings->mUpdatePosition = false;
	jolt_settings->mMakeRotationIdentity = false;
}

JoltSoftBodyImpl3D::~JoltSoftBodyImpl3D() {
	delete_safely(jolt_settings);
}

void JoltSoftBodyImpl3D::add_collision_exception(const RID& p_excepted_body) {
	exceptions.push_back(p_excepted_body);

	_exceptions_changed();
}

void JoltSoftBodyImpl3D::remove_collision_exception(const RID& p_excepted_body) {
	exceptions.erase(p_excepted_body);

	_exceptions_changed();
}

bool JoltSoftBodyImpl3D::has_collision_exception(const RID& p_excepted_body) const {
	return exceptions.find(p_excepted_body) >= 0;
}

TypedArray<RID> JoltSoftBodyImpl3D::get_collision_exceptions() const {
	TypedArray<RID> result;
	result.resize(exceptions.size());

	for (int32_t i = 0; i < exceptions.size(); ++i) {
		result[i] = exceptions[i];
	}

	return result;
}

bool JoltSoftBodyImpl3D::can_interact_with(const JoltBodyImpl3D& p_other) const {
	return (can_collide_with(p_other) || p_other.can_collide_with(*this)) &&
		!has_collision_exception(p_other.get_rid()) && !p_other.has_collision_exception(rid);
}

bool JoltSoftBodyImpl3D::can_interact_with(const JoltSoftBodyImpl3D& p_other) const {
	return (can_collide_with(p_other) || p_other.can_collide_with(*this)) &&
		!has_collision_exception(p_other.get_rid()) && !p_other.has_collision_exception(rid);
}

bool JoltSoftBodyImpl3D::can_interact_with(const JoltAreaImpl3D& p_other) const {
	return p_other.can_interact_with(*this);
}

Vector3 JoltSoftBodyImpl3D::get_velocity_at_position([[maybe_unused]] const Vector3& p_position
) const {
	return {0.0f, 0.0f, 0.0f};
}

void JoltSoftBodyImpl3D::set_mesh(const RID& p_mesh) {
	QUIET_FAIL_COND(mesh == p_mesh);

	mesh = p_mesh;

	_mesh_changed();
}

bool JoltSoftBodyImpl3D::is_sleeping() const {
	if (space == nullptr) {
		return false;
	}

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return !body->IsActive();
}

void JoltSoftBodyImpl3D::set_is_sleeping(bool p_enabled) {
	if (space == nullptr) {
		return;
	}

	JPH::BodyInterface& body_iface = space->get_body_iface();

	if (p_enabled) {
		body_iface.DeactivateBody(jolt_id);
	} else {
		body_iface.ActivateBody(jolt_id);
	}
}

void JoltSoftBodyImpl3D::set_simulation_precision(int32_t p_precision) {
	QUIET_FAIL_COND(simulation_precision == p_precision);

	simulation_precision = MAX(p_precision, 0);

	_update_simulation_precision();
}

void JoltSoftBodyImpl3D::set_mass(float p_mass) {
	QUIET_FAIL_COND(mass == p_mass);

	mass = MAX(p_mass, 0.0f);

	_update_mass();
}

float JoltSoftBodyImpl3D::get_stiffness_coefficient() const {
	return stiffness_coefficient;
}

void JoltSoftBodyImpl3D::set_stiffness_coefficient(float p_coefficient) {
	stiffness_coefficient = CLAMP(p_coefficient, 0.0f, 1.0f);
}

void JoltSoftBodyImpl3D::set_pressure(float p_pressure) {
	QUIET_FAIL_COND(pressure == p_pressure);

	pressure = MAX(p_pressure, 0.0f);

	_pressure_changed();
}

void JoltSoftBodyImpl3D::set_linear_damping(float p_damping) {
	QUIET_FAIL_COND(linear_damping == p_damping);

	linear_damping = MAX(p_damping, 0.0f);

	_update_damping();
}

float JoltSoftBodyImpl3D::get_drag() const {
	// Drag is not a thing in Jolt, and not supported by Godot Physics either.
	return 0.0f;
}

void JoltSoftBodyImpl3D::set_drag([[maybe_unused]] float p_drag) {
	// Drag is not a thing in Jolt, and not supported by Godot Physics either.
}

Variant JoltSoftBodyImpl3D::get_state(PhysicsServer3D::BodyState p_state) const {
	switch (p_state) {
		case PhysicsServer3D::BODY_STATE_TRANSFORM: {
			return get_transform();
		}
		case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY: {
			ERR_FAIL_D_NOT_IMPL();
		}
		case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY: {
			ERR_FAIL_D_NOT_IMPL();
		}
		case PhysicsServer3D::BODY_STATE_SLEEPING: {
			ERR_FAIL_D_NOT_IMPL();
		}
		case PhysicsServer3D::BODY_STATE_CAN_SLEEP: {
			ERR_FAIL_D_NOT_IMPL();
		}
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled body state: '%d'", p_state));
		}
	}
}

void JoltSoftBodyImpl3D::set_state(PhysicsServer3D::BodyState p_state, const Variant& p_value) {
	switch (p_state) {
		case PhysicsServer3D::BODY_STATE_TRANSFORM: {
			set_transform(p_value);
		} break;
		case PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY: {
			ERR_FAIL_NOT_IMPL();
		} break;
		case PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY: {
			ERR_FAIL_NOT_IMPL();
		} break;
		case PhysicsServer3D::BODY_STATE_SLEEPING: {
			ERR_FAIL_NOT_IMPL();
		} break;
		case PhysicsServer3D::BODY_STATE_CAN_SLEEP: {
			ERR_FAIL_NOT_IMPL();
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled body state: '%d'", p_state));
		} break;
	}
}

Transform3D JoltSoftBodyImpl3D::get_transform() const {
	// Since any transform gets baked into the vertices anyway we can just return identity here.
	return {};
}

void JoltSoftBodyImpl3D::set_transform(const Transform3D& p_transform) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to set transform for '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	// HACK(mihe): For whatever reason this has to be interpreted as a relative global-space
	// transform rather than an absolute one, because `SoftBody3D` will immediately upon entering
	// the scene tree set itself to be top-level and also set its transform to be identity, while
	// still expecting to stay in its original position.
	//
	// We also discard any scaling, since we have no way of scaling the actual edge lengths.
	const JPH::Mat44 relative_transform = to_jolt(p_transform.orthonormalized());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	JPH::Array<JPH::SoftBodyVertex>& physics_vertices = motion_properties.GetVertices();

	for (JPH::SoftBodyVertex& vertex : physics_vertices) {
		vertex.mPosition = vertex.mPreviousPosition = relative_transform * vertex.mPosition;
		vertex.mVelocity = JPH::Vec3::sZero();
	}
}

AABB JoltSoftBodyImpl3D::get_bounds() const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve world bounds of '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	return to_godot(body->GetWorldSpaceBounds());
}

void JoltSoftBodyImpl3D::update_rendering_server(
	PhysicsServer3DRenderingServerHandler* p_rendering_server_handler
) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to update rendering server with state of '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	const auto& motion_properties = static_cast<const JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	using SoftBodyVertex = JPH::SoftBodyMotionProperties::Vertex;
	using SoftBodyFace = JPH::SoftBodyMotionProperties::Face;

	const JPH::Array<SoftBodyVertex>& physics_vertices = motion_properties.GetVertices();
	const JPH::Array<SoftBodyFace>& physics_faces = motion_properties.GetFaces();

	const auto physics_vertex_count = (int32_t)physics_vertices.size();

	normals.resize(physics_vertex_count);

	for (const SoftBodyFace& physics_face : physics_faces) {
		// Jolt uses a different winding order, so we swap the indices to account for that.

		const uint32_t i0 = physics_face.mVertex[2];
		const uint32_t i1 = physics_face.mVertex[1];
		const uint32_t i2 = physics_face.mVertex[0];

		const Vector3 v0 = to_godot(physics_vertices[i0].mPosition);
		const Vector3 v1 = to_godot(physics_vertices[i1].mPosition);
		const Vector3 v2 = to_godot(physics_vertices[i2].mPosition);

		const Vector3 normal = (v2 - v0).cross(v1 - v0).normalized();

		normals[(int32_t)i0] = normal;
		normals[(int32_t)i1] = normal;
		normals[(int32_t)i2] = normal;
	}

	const int32_t mesh_vertex_count = shared->mesh_to_physics.size();

	for (int32_t i = 0; i < mesh_vertex_count; ++i) {
		const auto physics_index = (size_t)shared->mesh_to_physics[i];

		const Vector3 vertex = to_godot(physics_vertices[physics_index].mPosition);
		const Vector3 normal = normals[(int32_t)physics_index];

		p_rendering_server_handler->set_vertex(i, vertex);
		p_rendering_server_handler->set_normal(i, normal);
	}

	p_rendering_server_handler->set_aabb(get_bounds());
}

Vector3 JoltSoftBodyImpl3D::get_vertex_position(int32_t p_index) {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed to retrieve point position for '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	ERR_FAIL_INDEX_D(p_index, shared->mesh_to_physics.size());
	const int32_t physics_index = shared->mesh_to_physics[p_index];

	const JoltReadableBody3D body = space->read_body(jolt_id);
	ERR_FAIL_COND_D(body.is_invalid());

	const auto& motion_properties = static_cast<const JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	const JPH::Array<JPH::SoftBodyVertex>& physics_vertices = motion_properties.GetVertices();
	const JPH::SoftBodyVertex& physics_vertex = physics_vertices[(size_t)physics_index];

	return to_godot(body->GetCenterOfMassPosition() + physics_vertex.mPosition);
}

void JoltSoftBodyImpl3D::set_vertex_position(int32_t p_index, const Vector3& p_position) {
	ERR_FAIL_NULL_MSG(
		space,
		vformat(
			"Failed to set point position for '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	ERR_FAIL_INDEX(p_index, shared->mesh_to_physics.size());
	const int32_t physics_index = shared->mesh_to_physics[p_index];

	const float last_step = space->get_last_step();
	QUIET_FAIL_COND(last_step == 0.0f);

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	JPH::Array<JPH::SoftBodyVertex>& physics_vertices = motion_properties.GetVertices();
	JPH::SoftBodyVertex& physics_vertex = physics_vertices[(size_t)physics_index];

	const JPH::RVec3 center_of_mass = body->GetCenterOfMassPosition();
	const JPH::Vec3 local_position = JPH::Vec3(to_jolt_r(p_position) - center_of_mass);
	const JPH::Vec3 displacement = local_position - physics_vertex.mPosition;
	const JPH::Vec3 velocity = displacement / last_step;

	physics_vertex.mVelocity = velocity;
}

void JoltSoftBodyImpl3D::pin_vertex(int32_t p_index) {
	pinned_vertices.insert(p_index);

	_pins_changed();
}

void JoltSoftBodyImpl3D::unpin_vertex(int32_t p_index) {
	pinned_vertices.erase(p_index);

	_pins_changed();
}

void JoltSoftBodyImpl3D::unpin_all_vertices() {
	pinned_vertices.clear();

	_pins_changed();
}

bool JoltSoftBodyImpl3D::is_vertex_pinned(int32_t p_index) const {
	ERR_FAIL_NULL_D_MSG(
		space,
		vformat(
			"Failed retrieve pin status of point for '%s'. "
			"Doing so without a physics space is not supported by Godot Jolt. "
			"If this relates to a node, try adding the node to a scene tree first.",
			to_string()
		)
	);

	ERR_FAIL_INDEX_D(p_index, shared->mesh_to_physics.size());
	const int32_t physics_index = shared->mesh_to_physics[p_index];

	return pinned_vertices.has(physics_index);
}

String JoltSoftBodyImpl3D::to_string() const {
	Object* instance = ObjectDB::get_instance(instance_id);
	return instance != nullptr ? instance->to_string() : "<unknown>";
}

JPH::BroadPhaseLayer JoltSoftBodyImpl3D::_get_broad_phase_layer() const {
	return JoltBroadPhaseLayer::BODY_DYNAMIC;
}

JPH::ObjectLayer JoltSoftBodyImpl3D::_get_object_layer() const {
	ERR_FAIL_NULL_D(space);

	return space->map_to_object_layer(_get_broad_phase_layer(), collision_layer, collision_mask);
}

void JoltSoftBodyImpl3D::_space_changing() {
	JoltObjectImpl3D::_space_changing();

	_deref_shared_data();

	if (space != nullptr && !jolt_id.IsInvalid()) {
		const JoltWritableBody3D body = space->write_body(jolt_id);
		ERR_FAIL_COND(body.is_invalid());

		jolt_settings = new JPH::SoftBodyCreationSettings(body->GetSoftBodyCreationSettings());
		jolt_settings->mSettings = nullptr;
	}
}

void JoltSoftBodyImpl3D::_space_changed() {
	JoltObjectImpl3D::_space_changed();

	_update_mass();
	_update_pressure();
	_update_damping();
	_update_simulation_precision();
	_update_group_filter();
}

void JoltSoftBodyImpl3D::_add_to_space() {
	QUIET_FAIL_NULL(space);
	QUIET_FAIL_COND(!mesh.is_valid());

	ON_SCOPE_EXIT {
		delete_safely(jolt_settings);
	};

	const bool has_valid_shared = _ref_shared_data();
	ERR_FAIL_COND(!has_valid_shared);

	JPH::CollisionGroup::GroupID group_id = 0;
	JPH::CollisionGroup::SubGroupID sub_group_id = 0;
	JoltGroupFilter::encode_object(this, group_id, sub_group_id);

	jolt_settings->mSettings = shared->settings;
	jolt_settings->mUserData = reinterpret_cast<JPH::uint64>(this);
	jolt_settings->mObjectLayer = _get_object_layer();
	jolt_settings->mCollisionGroup = JPH::CollisionGroup(nullptr, group_id, sub_group_id);
	jolt_settings->mMaxLinearVelocity = JoltProjectSettings::get_max_linear_velocity();

	const JPH::BodyID new_jolt_id = space->add_soft_body(*this, *jolt_settings);
	QUIET_FAIL_COND(new_jolt_id.IsInvalid());

	jolt_id = new_jolt_id;
}

bool JoltSoftBodyImpl3D::_ref_shared_data() {
	using SoftBodyVertex = JPH::SoftBodySharedSettings::Vertex;
	using SoftBodyFace = JPH::SoftBodySharedSettings::Face;

	auto iter_shared_data = mesh_to_shared.find(mesh);

	if (iter_shared_data == mesh_to_shared.end()) {
		RenderingServer* rendering = RenderingServer::get_singleton();

		const Array mesh_data = rendering->mesh_surface_get_arrays(mesh, 0);
		ERR_FAIL_COND_D(mesh_data.is_empty());

		const PackedInt32Array mesh_indices = mesh_data[RenderingServer::ARRAY_INDEX];
		ERR_FAIL_COND_D(mesh_indices.is_empty());

		const PackedVector3Array mesh_vertices = mesh_data[RenderingServer::ARRAY_VERTEX];
		ERR_FAIL_COND_D(mesh_vertices.is_empty());

		iter_shared_data = mesh_to_shared.insert(mesh,Shared());

		JLocalVector<int32_t>& mesh_to_physics = iter_shared_data->second.mesh_to_physics;

		JPH::Ref<JPH::SoftBodySharedSettings>& settings = iter_shared_data->second.settings;
		settings->mVertexRadius = JoltProjectSettings::get_soft_body_point_margin();

		JPH::Array<SoftBodyVertex>& physics_vertices = settings->mVertices;
		JPH::Array<SoftBodyFace>& physics_faces = settings->mFaces;

		JHashMap<Vector3, int32_t> vertex_to_physics;

		const auto mesh_vertex_count = (int32_t)mesh_vertices.size();
		const auto mesh_index_count = (int32_t)mesh_indices.size();

		mesh_to_physics.resize(mesh_vertex_count);
		physics_vertices.reserve((size_t)mesh_vertex_count);
		vertex_to_physics.reserve(mesh_vertex_count);

		int32_t physics_index_count = 0;

		auto is_face_degenerate = [](const int32_t p_face[3]) {
			return p_face[0] == p_face[1] || p_face[0] == p_face[2] || p_face[1] == p_face[2];
		};

		for (int32_t i = 0; i < mesh_index_count; i += 3) {
			int32_t physics_face[3];
			int32_t mesh_face[3];

			for (int32_t j = 0; j < 3; ++j) {
				const int32_t mesh_index = mesh_indices[i + j];
				const Vector3 vertex = mesh_vertices[mesh_index];

				auto iter_physics_index = vertex_to_physics.find(vertex);

				if (iter_physics_index == vertex_to_physics.end()) {
					physics_vertices.emplace_back(
						JPH::Float3((float)vertex.x, (float)vertex.y, (float)vertex.z),
						JPH::Float3(0.0f, 0.0f, 0.0f),
						1.0f
					);

					iter_physics_index = vertex_to_physics.insert(vertex, physics_index_count++);
				}

				mesh_face[j] = mesh_index;
				physics_face[j] = iter_physics_index->second;
				mesh_to_physics[mesh_index] = iter_physics_index->second;
			}

			ERR_CONTINUE_MSG(
				is_face_degenerate(physics_face),
				vformat(
					"Failed to append face to soft body '%s'. "
					"Face was found to be degenerate. "
					"Face consist of indices %d, %d and %d.",
					to_string(),
					mesh_face[0],
					mesh_face[1],
					mesh_face[2]
				)
			);

			// Jolt uses a different winding order, so we swap the indices to account for that.

			physics_faces.emplace_back(
				(JPH::uint32)physics_face[2],
				(JPH::uint32)physics_face[1],
				(JPH::uint32)physics_face[0]
			);
		}

		// Pin the static vertices, this is used during the Optimize call to order the constraints.
		// Note that it is ok if the pinned vertices change later, this will reduce the
		// effectiveness of the constraints a bit.
		for (int32_t pin_mesh_index : pinned_vertices) {
			ERR_FAIL_INDEX_V(pin_mesh_index, mesh_to_physics.size(), false);
			const int32_t pin_physics_index = mesh_to_physics[pin_mesh_index];
			ERR_FAIL_INDEX_V(pin_physics_index, (int32_t)physics_vertices.size(), false);

			physics_vertices[(size_t)pin_physics_index].mInvMass = 0.0f;
		}

		// HACK(mihe): Since Godot's stiffness is input as a coefficient between 0 and 1, and Jolt
		// uses actual stiffness for its edge constraints, we crudely map one to the other with an
		// arbitrary constant.
		const float stiffness = MAX(Math::pow(stiffness_coefficient, 3.0f) * 100000.0f, 0.000001f);
		const float inverse_stiffness = 1.0f / stiffness;

		JPH::SoftBodySharedSettings::VertexAttributes vertex_attrib;
		vertex_attrib.mCompliance = vertex_attrib.mShearCompliance = inverse_stiffness;

		settings->CreateConstraints(&vertex_attrib, 1, JPH::SoftBodySharedSettings::EBendType::None);
		settings->Optimize();
	} else {
		iter_shared_data->second.ref_count++;
	}

	shared = &iter_shared_data->second;

	return true;
}

void JoltSoftBodyImpl3D::_deref_shared_data() {
	QUIET_FAIL_NULL(shared);

	auto iter = mesh_to_shared.find(mesh);
	QUIET_FAIL_COND(iter == mesh_to_shared.end());

	if (--iter->second.ref_count == 0) {
		mesh_to_shared.remove(iter);
	}

	shared = nullptr;
}

void JoltSoftBodyImpl3D::_update_mass() {
	QUIET_FAIL_NULL(space);
	QUIET_FAIL_COND(jolt_id.IsInvalid());

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	JPH::Array<JPH::SoftBodyVertex>& physics_vertices = motion_properties.GetVertices();

	const float inverse_vertex_mass = mass == 0.0f ? 1.0f : (float)physics_vertices.size() / mass;

	for (JPH::SoftBodyVertex& vertex : physics_vertices) {
		vertex.mInvMass = inverse_vertex_mass;
	}

	for (int32_t pin_mesh_index : pinned_vertices) {
		ERR_FAIL_INDEX(pin_mesh_index, shared->mesh_to_physics.size());
		const int32_t pin_physics_index = shared->mesh_to_physics[pin_mesh_index];
		ERR_FAIL_INDEX(pin_physics_index, (int32_t)physics_vertices.size());

		physics_vertices[(size_t)pin_physics_index].mInvMass = 0.0f;
	}
}

void JoltSoftBodyImpl3D::_update_pressure() {
	if (space == nullptr) {
		jolt_settings->mPressure = pressure;
		return;
	}

	QUIET_FAIL_COND(jolt_id.IsInvalid());

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	motion_properties.SetPressure(pressure);
}

void JoltSoftBodyImpl3D::_update_damping() {
	if (space == nullptr) {
		jolt_settings->mLinearDamping = linear_damping;
		return;
	}

	QUIET_FAIL_COND(jolt_id.IsInvalid());

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	motion_properties.SetLinearDamping(linear_damping);
}

void JoltSoftBodyImpl3D::_update_simulation_precision() {
	if (space == nullptr) {
		jolt_settings->mNumIterations = (JPH::uint32)simulation_precision;
		return;
	}

	QUIET_FAIL_COND(jolt_id.IsInvalid());

	JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	auto& motion_properties = static_cast<JPH::SoftBodyMotionProperties&>(
		*body->GetMotionPropertiesUnchecked()
	);

	motion_properties.SetNumIterations((JPH::uint32)simulation_precision);
}

void JoltSoftBodyImpl3D::_update_group_filter() {
	JPH::GroupFilter* group_filter = !exceptions.is_empty() ? JoltGroupFilter::instance : nullptr;

	if (space == nullptr) {
		jolt_settings->mCollisionGroup.SetGroupFilter(group_filter);
		return;
	}

	QUIET_FAIL_COND(jolt_id.IsInvalid());

	const JoltWritableBody3D body = space->write_body(jolt_id);
	ERR_FAIL_COND(body.is_invalid());

	body->GetCollisionGroup().SetGroupFilter(group_filter);
}

void JoltSoftBodyImpl3D::_try_rebuild() {
	if (space != nullptr) {
		_deref_shared_data();
		_reset_space();
	}
}

void JoltSoftBodyImpl3D::_mesh_changed() {
	_try_rebuild();
}

void JoltSoftBodyImpl3D::_pressure_changed() {
	_update_pressure();
	wake_up();
}

void JoltSoftBodyImpl3D::_damping_changed() {
	_update_damping();
	wake_up();
}

void JoltSoftBodyImpl3D::_pins_changed() {
	_update_mass();
	wake_up();
}

void JoltSoftBodyImpl3D::_exceptions_changed() {
	_update_group_filter();
}
