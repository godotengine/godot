/**************************************************************************/
/*  jolt_soft_body_3d.cpp                                                 */
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

#include "jolt_soft_body_3d.h"

#include "../jolt_project_settings.h"
#include "../misc/jolt_type_conversions.h"
#include "../spaces/jolt_broad_phase_layer.h"
#include "../spaces/jolt_space_3d.h"
#include "jolt_area_3d.h"
#include "jolt_body_3d.h"
#include "jolt_group_filter.h"

#include "core/config/engine.h"
#include "servers/physics_3d/physics_server_3d_rendering_server_handler.h"
#include "servers/rendering/rendering_server.h"

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

namespace {

template <typename TJoltVertex>
void pin_vertices(const JoltSoftBody3D &p_body, const HashSet<int> &p_pinned_vertices, const HashMap<int, float> &p_pinned_weights, const LocalVector<int> &p_mesh_to_physics, const HashMap<int, int> &p_mesh_to_anchor, JPH::Array<TJoltVertex> &r_physics_vertices, float p_mass) {
	const int mesh_vertex_count = p_mesh_to_physics.size();
	const int physics_vertex_count = (int)r_physics_vertices.size();
	const float default_inv_mass = (physics_vertex_count > 0 && p_mass > 0.0f) ? (float)physics_vertex_count / p_mass : 1.0f;

	for (int i = 0; i < physics_vertex_count; ++i) {
		r_physics_vertices[i].mInvMass = default_inv_mass;
	}

	for (const KeyValue<int, int> &E : p_mesh_to_anchor) {
		if (E.value >= 0 && E.value < physics_vertex_count) {
			r_physics_vertices[E.value].mInvMass = 0.0f;
		}
	}

	for (int mesh_index : p_pinned_vertices) {
		ERR_CONTINUE_MSG(mesh_index < 0 || mesh_index >= mesh_vertex_count, vformat("Index %d of pinned vertex in soft body '%s' is out of bounds. There are only %d vertices in the current mesh.", mesh_index, p_body.to_string(), mesh_vertex_count));

		const int physics_index = p_mesh_to_physics[mesh_index];
		ERR_CONTINUE_MSG(physics_index < 0 || physics_index >= physics_vertex_count, vformat("Index %d of pinned vertex in soft body '%s' is out of bounds. There are only %d vertices in the current mesh. This should not happen. Please report this.", physics_index, p_body.to_string(), physics_vertex_count));

		float weight = 1.0f;
		const HashMap<int, float>::ConstIterator E = p_pinned_weights.find(mesh_index);
		if (E) {
			weight = E->value;
		}

		if (weight >= 0.999f) {
			r_physics_vertices[physics_index].mInvMass = 0.0f;
		}
	}
}

} // namespace

JPH::BroadPhaseLayer JoltSoftBody3D::_get_broad_phase_layer() const {
	return JoltBroadPhaseLayer::BODY_DYNAMIC;
}

JPH::ObjectLayer JoltSoftBody3D::_get_object_layer() const {
	ERR_FAIL_NULL_V(space, 0);

	return space->map_to_object_layer(_get_broad_phase_layer(), collision_layer, collision_mask);
}

void JoltSoftBody3D::_space_changing() {
	JoltObject3D::_space_changing();

	if (in_space()) {
		jolt_settings = new JPH::SoftBodyCreationSettings(jolt_body->GetSoftBodyCreationSettings());
		jolt_settings->mSettings = nullptr;
		jolt_settings->mVertexRadius = JoltProjectSettings::soft_body_point_radius;
	}
}

void JoltSoftBody3D::_space_changed() {
	JoltObject3D::_space_changed();

	_update_mass();
	_update_pressure();
	_update_damping();
	_update_simulation_precision();
	_update_group_filter();
}

void JoltSoftBody3D::_add_to_space() {
	if (unlikely(space == nullptr || !mesh.is_valid())) {
		return;
	}

	JPH::SoftBodySharedSettings *shared_settings = _create_shared_settings();
	ERR_FAIL_NULL(shared_settings);

	JPH::CollisionGroup::GroupID group_id = 0;
	JPH::CollisionGroup::SubGroupID sub_group_id = 0;
	JoltGroupFilter::encode_object(this, group_id, sub_group_id);

	jolt_settings->mSettings = shared_settings;
	jolt_settings->mUserData = reinterpret_cast<JPH::uint64>(this);
	jolt_settings->mObjectLayer = _get_object_layer();
	jolt_settings->mCollisionGroup = JPH::CollisionGroup(nullptr, group_id, sub_group_id);
	jolt_settings->mMaxLinearVelocity = JoltProjectSettings::max_linear_velocity;

	JPH::Body *new_jolt_body = space->add_object(*this, *jolt_settings);
	if (new_jolt_body == nullptr) {
		return;
	}

	jolt_body = new_jolt_body;

	delete jolt_settings;
	jolt_settings = nullptr;
}

JPH::SoftBodySharedSettings *JoltSoftBody3D::_create_shared_settings() {
	RenderingServer *rendering = RenderingServer::get_singleton();

	// TODO: calling RenderingServer::mesh_surface_get_arrays() from the physics thread
	// is not safe and can deadlock when physics/3d/run_on_separate_thread is enabled.
	// This method blocks on the main thread to return data, but the main thread may be
	// blocked waiting on us in PhysicsServer3D::sync().

	const int surface_count = rendering->mesh_get_surface_count(mesh);
	ERR_FAIL_COND_V(surface_count <= 0, nullptr);

	JPH::SoftBodySharedSettings *settings = new JPH::SoftBodySharedSettings();
	JPH::Array<JPH::SoftBodySharedSettings::Vertex> &physics_vertices = settings->mVertices;
	JPH::Array<JPH::SoftBodySharedSettings::Face> &physics_faces = settings->mFaces;

	HashMap<Vector3, int> vertex_to_physics;

	int total_mesh_vertices = 0;
	for (int s = 0; s < surface_count; s++) {
		const Array surface_arrays = rendering->mesh_surface_get_arrays(mesh, s);
		if (!surface_arrays.is_empty() && surface_arrays.size() > RSE::ARRAY_VERTEX) {
			const PackedVector3Array verts = surface_arrays[RSE::ARRAY_VERTEX];
			total_mesh_vertices += verts.size();
		}
	}
	ERR_FAIL_COND_V(total_mesh_vertices == 0, nullptr);

	mesh_to_physics.resize(total_mesh_vertices);
	for (int &index : mesh_to_physics) {
		index = -1;
	}
	physics_vertices.reserve(total_mesh_vertices);
	vertex_to_physics.reserve(total_mesh_vertices);

	int physics_index_count = 0;
	const JPH::RVec3 body_position = jolt_settings->mPosition;

	struct LooseEdge {
		int v0;
		int v1;
	};
	LocalVector<LooseEdge> loose_edges;

	int global_vertex_offset = 0;

	for (int s = 0; s < surface_count; s++) {
		const Array surface_arrays = rendering->mesh_surface_get_arrays(mesh, s);
		if (surface_arrays.is_empty() || surface_arrays.size() <= RSE::ARRAY_VERTEX) {
			continue;
		}

		const PackedVector3Array mesh_vertices = surface_arrays[RSE::ARRAY_VERTEX];
		PackedInt32Array mesh_indices;
		if (surface_arrays.size() > RSE::ARRAY_INDEX) {
			mesh_indices = surface_arrays[RSE::ARRAY_INDEX];
		}
		const int mesh_vertex_count = mesh_vertices.size();
		const int mesh_index_count = mesh_indices.size();

		RenderingServerTypes::SurfaceData surface_data = rendering->mesh_get_surface(mesh, s);
		RSE::PrimitiveType prim_type = surface_data.primitive;

		for (int v = 0; v < mesh_vertex_count; ++v) {
			const Vector3 vertex = mesh_vertices[v];
			HashMap<Vector3, int>::Iterator iter_physics_index = vertex_to_physics.find(vertex);

			if (iter_physics_index == vertex_to_physics.end()) {
				physics_vertices.emplace_back(JPH::Float3((float)(vertex.x - body_position.GetX()), (float)(vertex.y - body_position.GetY()), (float)(vertex.z - body_position.GetZ())), JPH::Float3(0.0f, 0.0f, 0.0f), 1.0f);
				iter_physics_index = vertex_to_physics.insert(vertex, physics_index_count++);
			}

			mesh_to_physics[global_vertex_offset + v] = iter_physics_index->value;
		}

		if (prim_type == RSE::PRIMITIVE_TRIANGLES) {
			if (!mesh_indices.is_empty()) {
				for (int i = 0; i < mesh_index_count; i += 3) {
					int physics_face[3];
					for (int j = 0; j < 3; ++j) {
						int local_idx = mesh_indices[i + j];
						physics_face[j] = mesh_to_physics[global_vertex_offset + local_idx];
					}

					if (physics_face[0] == physics_face[1] || physics_face[0] == physics_face[2] || physics_face[1] == physics_face[2]) {
						continue;
					}

					physics_faces.emplace_back((JPH::uint32)physics_face[2], (JPH::uint32)physics_face[1], (JPH::uint32)physics_face[0]);
				}
			} else {
				for (int i = 0; i + 2 < mesh_vertex_count; i += 3) {
					int physics_face[3];
					for (int j = 0; j < 3; ++j) {
						physics_face[j] = mesh_to_physics[global_vertex_offset + i + j];
					}

					if (physics_face[0] == physics_face[1] || physics_face[0] == physics_face[2] || physics_face[1] == physics_face[2]) {
						continue;
					}

					physics_faces.emplace_back((JPH::uint32)physics_face[2], (JPH::uint32)physics_face[1], (JPH::uint32)physics_face[0]);
				}
			}
		} else if (prim_type == RSE::PRIMITIVE_LINES) {
			if (!mesh_indices.is_empty()) {
				for (int i = 0; i + 1 < mesh_index_count; i += 2) {
					int p0 = mesh_to_physics[global_vertex_offset + mesh_indices[i]];
					int p1 = mesh_to_physics[global_vertex_offset + mesh_indices[i + 1]];
					if (p0 != p1) {
						loose_edges.push_back({ p0, p1 });
					}
				}
			} else {
				for (int i = 0; i + 1 < mesh_vertex_count; i += 2) {
					int p0 = mesh_to_physics[global_vertex_offset + i];
					int p1 = mesh_to_physics[global_vertex_offset + i + 1];
					if (p0 != p1) {
						loose_edges.push_back({ p0, p1 });
					}
				}
			}
		} else if (prim_type == RSE::PRIMITIVE_LINE_STRIP) {
			if (!mesh_indices.is_empty()) {
				for (int i = 0; i + 1 < mesh_index_count; ++i) {
					int p0 = mesh_to_physics[global_vertex_offset + mesh_indices[i]];
					int p1 = mesh_to_physics[global_vertex_offset + mesh_indices[i + 1]];
					if (p0 != p1) {
						loose_edges.push_back({ p0, p1 });
					}
				}
			} else {
				for (int i = 0; i + 1 < mesh_vertex_count; ++i) {
					int p0 = mesh_to_physics[global_vertex_offset + i];
					int p1 = mesh_to_physics[global_vertex_offset + i + 1];
					if (p0 != p1) {
						loose_edges.push_back({ p0, p1 });
					}
				}
			}
		}

		global_vertex_offset += mesh_vertex_count;
	}

	mesh_to_anchor.clear();

	const float w1_plus_w2 = 2.0f * physics_vertices.size() / mass;
	const float dt = 1.0f / Engine::get_singleton()->get_user_physics_ticks_per_second() / simulation_precision;
	const float inverse_stiffness = dt * dt * (1.0f / stiffness_coefficient - 1.0f) * w1_plus_w2;

	JPH::SoftBodySharedSettings::VertexAttributes vertex_attrib;
	vertex_attrib.mCompliance = vertex_attrib.mShearCompliance = inverse_stiffness;

	settings->CreateConstraints(&vertex_attrib, 1, JPH::SoftBodySharedSettings::EBendType::None);

	float multiplier = 1.0f - shrinking_factor;
	for (JPH::SoftBodySharedSettings::Edge &e : settings->mEdgeConstraints) {
		e.mRestLength *= multiplier;
	}

	// Add internal springs from loose edges only when enabled
	if (internal_springs && !loose_edges.is_empty()) {
		const float internal_stiffness = CLAMP(internal_spring_stiffness, 0.0001f, 1.0f);
		const float internal_compliance = dt * dt * (1.0f / internal_stiffness - 1.0f) * w1_plus_w2;
		for (const LooseEdge &le : loose_edges) {
			const JPH::Float3 &pos0 = physics_vertices[le.v0].mPosition;
			const JPH::Float3 &pos1 = physics_vertices[le.v1].mPosition;
			float dx = pos0.x - pos1.x;
			float dy = pos0.y - pos1.y;
			float dz = pos0.z - pos1.z;
			float rest_len = Math::sqrt(dx * dx + dy * dy + dz * dz) * multiplier;

			JPH::SoftBodySharedSettings::Edge edge_constraint((JPH::uint32)le.v0, (JPH::uint32)le.v1, internal_compliance);
			edge_constraint.mRestLength = rest_len;
			settings->mEdgeConstraints.push_back(edge_constraint);
		}
	}

	// Add anchor vertices and spring edges for soft-pinned points
	for (int mesh_index : pinned_vertices) {
		if (mesh_index < 0 || mesh_index >= (int)mesh_to_physics.size()) {
			continue;
		}
		const int mesh_physics_index = mesh_to_physics[mesh_index];
		if (mesh_physics_index < 0 || mesh_physics_index >= (int)physics_vertices.size()) {
			continue;
		}

		float weight = 1.0f;
		const HashMap<int, float>::ConstIterator E = pinned_weights.find(mesh_index);
		if (E) {
			weight = E->value;
		}

		if (weight > 0.0f && weight < 0.999f) {
			int anchor_physics_index = (int)physics_vertices.size();
			const JPH::Float3 &mesh_pos = physics_vertices[mesh_physics_index].mPosition;
			physics_vertices.emplace_back(mesh_pos, JPH::Float3(0.0f, 0.0f, 0.0f), 0.0f);
			mesh_to_anchor[mesh_index] = anchor_physics_index;

			const float spring_stiffness = CLAMP(weight, 0.0001f, 1.0f);
			const float spring_compliance = dt * dt * (1.0f / spring_stiffness - 1.0f) * w1_plus_w2;

			JPH::SoftBodySharedSettings::Edge anchor_edge((JPH::uint32)mesh_physics_index, (JPH::uint32)anchor_physics_index, spring_compliance);
			anchor_edge.mRestLength = 0.0f;
			settings->mEdgeConstraints.push_back(anchor_edge);
		}
	}

	settings->mVertices = physics_vertices;
	settings->Optimize();
	pin_vertices(*this, pinned_vertices, pinned_weights, mesh_to_physics, mesh_to_anchor, settings->mVertices, mass);

	return settings;
}

void JoltSoftBody3D::_apply_environmental_forces(float p_step) {
	// Get approximation of the center of the soft body.
	Vector3 com_position = to_godot(jolt_body->GetCenterOfMassPosition());

	// Calculate gravity and which areas affect the soft body through wind.
	bool gravity_done = false;
	Vector3 gravity;
	LocalVector<JoltArea3D *> wind_areas;
	for (JoltArea3D *area : areas) {
		if (!gravity_done) {
			gravity_done = JoltArea3D::apply_override(gravity, area->get_gravity_mode(), [&]() {
				return area->compute_gravity(com_position);
			});
		}

		if (area->get_wind_pressure() > CMP_EPSILON) {
			wind_areas.push_back(area);
		}
	}

	// Add default gravity.
	if (!gravity_done) {
		gravity += space->get_default_area()->compute_gravity(com_position);
	}

	// Apply gravity to soft body. Note that this only works so long as vertices have uniform mass (excluding pinned vertices).
	jolt_body->AddForce(to_jolt(gravity) * mass);

	if (!wind_areas.is_empty()) {
		JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
		JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();

		for (const JPH::SoftBodySharedSettings::Face &physics_face : motion_properties.GetFaces()) {
			JPH::SoftBodyVertex &physics_vertex0 = physics_vertices[physics_face.mVertex[0]];
			JPH::SoftBodyVertex &physics_vertex1 = physics_vertices[physics_face.mVertex[1]];
			JPH::SoftBodyVertex &physics_vertex2 = physics_vertices[physics_face.mVertex[2]];

			// Calculate the triangle centroid.
			Vector3 v0 = to_godot(physics_vertex0.mPosition);
			Vector3 v1 = to_godot(physics_vertex1.mPosition);
			Vector3 v2 = to_godot(physics_vertex2.mPosition);
			Vector3 centroid = com_position + (v0 + v1 + v2) * real_t(1.0 / 3.0);

			// Calculate the triangle normal.
			Vector3 normal = (v2 - v0).cross(v1 - v0);
			real_t normal_length = normal.length();
			if (normal_length > real_t(1.0e-6)) { // If the normal is near zero, the area is near zero so we can skip this triangle.
				normal /= normal_length;

				// Area is half the length of the cross product of two sides.
				real_t triangle_area = real_t(0.5) * normal_length;

				// Accumulate wind forces from all wind areas.
				Vector3 wind_force;
				for (const JoltArea3D *area : wind_areas) {
					const Vector3 &wind_direction = area->get_wind_direction();
					const Vector3 &wind_source = area->get_wind_source();

					// Calculate attenuation factor based on distance from wind source to triangle centroid.
					// We do not allow a projection below 1 to ensure that we never amplify and to avoid NaNs when the value would be negative.
					real_t projection_toward_centroid = MAX((centroid - wind_source).dot(wind_direction), real_t(1.0));
					real_t attenuation_over_distance = Math::pow(projection_toward_centroid, -real_t(area->get_wind_attenuation_factor()));

					// Calculate force magnitude.
					real_t force_magnitude = area->get_wind_pressure() * triangle_area;

					// Calculate the resulting wind force on the triangle by projecting wind direction onto triangle normal.
					// Divide by 3 to distribute force equally over each vertex.
					wind_force += (force_magnitude * attenuation_over_distance * real_t(1.0 / 3.0) * normal.dot(wind_direction)) * normal;
				}

				// Apply the force as an impulse over the timestep.
				JPH::Vec3 impulse = to_jolt(wind_force * p_step);
				physics_vertex0.mVelocity += impulse * physics_vertex0.mInvMass;
				physics_vertex1.mVelocity += impulse * physics_vertex1.mInvMass;
				physics_vertex2.mVelocity += impulse * physics_vertex2.mInvMass;
			}
		}
	}
}

void JoltSoftBody3D::_update_mass() {
	if (!in_space()) {
		return;
	}

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();

	const float inverse_vertex_mass = (float)physics_vertices.size() / mass;

	for (JPH::SoftBodyVertex &vertex : physics_vertices) {
		vertex.mInvMass = inverse_vertex_mass;
	}

	pin_vertices(*this, pinned_vertices, pinned_weights, mesh_to_physics, mesh_to_anchor, physics_vertices, mass);
}

void JoltSoftBody3D::_update_pressure() {
	if (!in_space()) {
		jolt_settings->mPressure = pressure;
		return;
	}

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	motion_properties.SetPressure(pressure);
}

void JoltSoftBody3D::_update_damping() {
	if (!in_space()) {
		jolt_settings->mLinearDamping = linear_damping;
		return;
	}

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	motion_properties.SetLinearDamping(linear_damping);
}

void JoltSoftBody3D::_update_simulation_precision() {
	if (!in_space()) {
		jolt_settings->mNumIterations = (JPH::uint32)simulation_precision;
		return;
	}

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	motion_properties.SetNumIterations((JPH::uint32)simulation_precision);
}

void JoltSoftBody3D::_update_group_filter() {
	JPH::GroupFilter *group_filter = !exceptions.is_empty() ? JoltGroupFilter::instance : nullptr;

	if (!in_space()) {
		jolt_settings->mCollisionGroup.SetGroupFilter(group_filter);
	} else {
		jolt_body->GetCollisionGroup().SetGroupFilter(group_filter);
	}
}

void JoltSoftBody3D::_try_rebuild() {
	if (space != nullptr) {
		_reset_space();
	}
}

void JoltSoftBody3D::_mesh_changed() {
	_try_rebuild();
}

void JoltSoftBody3D::_simulation_precision_changed() {
	wake_up();
}

void JoltSoftBody3D::_mass_changed() {
	_update_mass();
	wake_up();
}

void JoltSoftBody3D::_pressure_changed() {
	_update_pressure();
	wake_up();
}

void JoltSoftBody3D::_damping_changed() {
	_update_damping();
	wake_up();
}

void JoltSoftBody3D::_pins_changed() {
	_update_mass();
	wake_up();
}

void JoltSoftBody3D::_vertices_changed() {
	wake_up();
}

void JoltSoftBody3D::_exceptions_changed() {
	_update_group_filter();
}

void JoltSoftBody3D::_motion_changed() {
	wake_up();
}

void JoltSoftBody3D::_transform_changed() {
	wake_up();
}

void JoltSoftBody3D::_areas_changed() {
	wake_up();
}

JoltSoftBody3D::JoltSoftBody3D() :
		JoltObject3D(OBJECT_TYPE_SOFT_BODY) {
	jolt_settings->mRestitution = 0.0f;
	jolt_settings->mFriction = 1.0f;
	jolt_settings->mUpdatePosition = true;
	jolt_settings->mMakeRotationIdentity = false;
}

JoltSoftBody3D::~JoltSoftBody3D() {
	if (jolt_settings != nullptr) {
		delete jolt_settings;
		jolt_settings = nullptr;
	}
}

void JoltSoftBody3D::add_collision_exception(const RID &p_excepted_body) {
	exceptions.push_back(p_excepted_body);

	_exceptions_changed();
}

void JoltSoftBody3D::remove_collision_exception(const RID &p_excepted_body) {
	exceptions.erase(p_excepted_body);

	_exceptions_changed();
}

bool JoltSoftBody3D::has_collision_exception(const RID &p_excepted_body) const {
	return exceptions.find(p_excepted_body) >= 0;
}

void JoltSoftBody3D::add_area(JoltArea3D *p_area) {
	int i = 0;
	for (; i < (int)areas.size(); i++) {
		if (p_area->get_priority() > areas[i]->get_priority()) {
			break;
		}
	}

	areas.insert(i, p_area);

	_areas_changed();
}

void JoltSoftBody3D::remove_area(JoltArea3D *p_area) {
	areas.erase(p_area);

	_areas_changed();
}

bool JoltSoftBody3D::can_interact_with(const JoltBody3D &p_other) const {
	return (can_collide_with(p_other) || p_other.can_collide_with(*this)) && !has_collision_exception(p_other.get_rid()) && !p_other.has_collision_exception(rid);
}

bool JoltSoftBody3D::can_interact_with(const JoltSoftBody3D &p_other) const {
	return (can_collide_with(p_other) || p_other.can_collide_with(*this)) && !has_collision_exception(p_other.get_rid()) && !p_other.has_collision_exception(rid);
}

bool JoltSoftBody3D::can_interact_with(const JoltArea3D &p_other) const {
	return p_other.can_interact_with(*this);
}

Vector3 JoltSoftBody3D::get_velocity_at_position(const Vector3 &p_position) const {
	return Vector3();
}

void JoltSoftBody3D::pre_step(float p_step) {
	_apply_environmental_forces(p_step);
}

void JoltSoftBody3D::set_mesh(const RID &p_mesh) {
	if (unlikely(mesh == p_mesh)) {
		return;
	}

	mesh = p_mesh;
	_mesh_changed();
}

bool JoltSoftBody3D::is_sleeping() const {
	if (!in_space()) {
		return false;
	} else {
		return !jolt_body->IsActive();
	}
}

void JoltSoftBody3D::apply_vertex_impulse(int p_index, const Vector3 &p_impulse) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to apply impulse to '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	ERR_FAIL_INDEX(p_index, (int)mesh_to_physics.size());
	const int physics_index = mesh_to_physics[p_index];
	ERR_FAIL_COND_MSG(physics_index < 0, vformat("Soft body vertex %d was not used by a face and has been omitted for '%s'. No impulse can be applied.", p_index, to_string()));
	ERR_FAIL_COND_MSG(pinned_vertices.has(physics_index), vformat("Failed to apply impulse to point at index %d for '%s'. Point was found to be pinned.", static_cast<int>(physics_index), to_string()));

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());

	JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();
	JPH::SoftBodyVertex &physics_vertex = physics_vertices[physics_index];

	physics_vertex.mVelocity += to_jolt(p_impulse) * physics_vertex.mInvMass;

	_motion_changed();
}

void JoltSoftBody3D::apply_vertex_force(int p_index, const Vector3 &p_force) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to apply force to '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	apply_vertex_impulse(p_index, p_force * space->get_last_step());
}

void JoltSoftBody3D::apply_central_impulse(const Vector3 &p_impulse) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to apply central impulse to '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();

	const JPH::Vec3 impulse = to_jolt(p_impulse) / physics_vertices.size();

	for (JPH::SoftBodyVertex &physics_vertex : physics_vertices) {
		if (physics_vertex.mInvMass > 0.0f) {
			physics_vertex.mVelocity += impulse * physics_vertex.mInvMass;
		}
	}

	_motion_changed();
}

void JoltSoftBody3D::apply_central_force(const Vector3 &p_force) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to apply central force to '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	jolt_body->AddForce(to_jolt(p_force));

	_motion_changed();
}

void JoltSoftBody3D::set_is_sleeping(bool p_enabled) {
	if (!in_space()) {
		return;
	}

	space->set_is_object_sleeping(jolt_body->GetID(), p_enabled);
}

bool JoltSoftBody3D::is_sleep_allowed() const {
	if (!in_space()) {
		return jolt_settings->mAllowSleeping;
	} else {
		return jolt_body->GetAllowSleeping();
	}
}

void JoltSoftBody3D::set_is_sleep_allowed(bool p_enabled) {
	if (!in_space()) {
		jolt_settings->mAllowSleeping = p_enabled;
	} else {
		jolt_body->SetAllowSleeping(p_enabled);
	}
}

void JoltSoftBody3D::set_simulation_precision(int p_precision) {
	if (unlikely(simulation_precision == p_precision)) {
		return;
	}

	simulation_precision = MAX(p_precision, 0);

	_simulation_precision_changed();
}

void JoltSoftBody3D::set_mass(float p_mass) {
	ERR_FAIL_COND(p_mass <= 0.0); // A mass of zero would result in infinite inverse mass.

	if (unlikely(mass == p_mass)) {
		return;
	}

	mass = p_mass;

	_mass_changed();
}

float JoltSoftBody3D::get_stiffness_coefficient() const {
	return stiffness_coefficient;
}

void JoltSoftBody3D::set_stiffness_coefficient(float p_coefficient) {
	stiffness_coefficient = CLAMP(p_coefficient, 0.0f, 1.0f);
}

void JoltSoftBody3D::set_internal_springs(bool p_enabled) {
	if (internal_springs == p_enabled) {
		return;
	}
	internal_springs = p_enabled;
	_try_rebuild();
}

void JoltSoftBody3D::set_internal_spring_stiffness(float p_stiffness) {
	internal_spring_stiffness = CLAMP(p_stiffness, 0.0f, 1.0f);
	_try_rebuild();
}

float JoltSoftBody3D::get_shrinking_factor() const {
	return shrinking_factor;
}

void JoltSoftBody3D::set_shrinking_factor(float p_shrinking_factor) {
	shrinking_factor = p_shrinking_factor;
}

void JoltSoftBody3D::set_pressure(float p_pressure) {
	if (unlikely(pressure == p_pressure)) {
		return;
	}

	pressure = MAX(p_pressure, 0.0f);

	_pressure_changed();
}

void JoltSoftBody3D::set_linear_damping(float p_damping) {
	if (unlikely(linear_damping == p_damping)) {
		return;
	}

	linear_damping = MAX(p_damping, 0.0f);

	_damping_changed();
}

float JoltSoftBody3D::get_drag() const {
	// Drag is not a thing in Jolt, and not supported by Godot Physics either.
	return 0.0f;
}

void JoltSoftBody3D::set_drag(float p_drag) {
	// Drag is not a thing in Jolt, and not supported by Godot Physics either.
}

Variant JoltSoftBody3D::get_state(PS3DE::BodyState p_state) const {
	switch (p_state) {
		case PS3DE::BODY_STATE_TRANSFORM: {
			return get_transform();
		}
		case PS3DE::BODY_STATE_LINEAR_VELOCITY: {
			ERR_FAIL_V_MSG(Variant(), "Linear velocity is not supported for soft bodies.");
		}
		case PS3DE::BODY_STATE_ANGULAR_VELOCITY: {
			ERR_FAIL_V_MSG(Variant(), "Angular velocity is not supported for soft bodies.");
		}
		case PS3DE::BODY_STATE_SLEEPING: {
			return is_sleeping();
		}
		case PS3DE::BODY_STATE_CAN_SLEEP: {
			return is_sleep_allowed();
		}
		default: {
			ERR_FAIL_V_MSG(Variant(), vformat("Unhandled body state: '%d'. This should not happen. Please report this.", p_state));
		}
	}
}

void JoltSoftBody3D::set_state(PS3DE::BodyState p_state, const Variant &p_value) {
	switch (p_state) {
		case PS3DE::BODY_STATE_TRANSFORM: {
			set_transform(p_value);
		} break;
		case PS3DE::BODY_STATE_LINEAR_VELOCITY: {
			ERR_FAIL_MSG("Linear velocity is not supported for soft bodies.");
		} break;
		case PS3DE::BODY_STATE_ANGULAR_VELOCITY: {
			ERR_FAIL_MSG("Angular velocity is not supported for soft bodies.");
		} break;
		case PS3DE::BODY_STATE_SLEEPING: {
			set_is_sleeping(p_value);
		} break;
		case PS3DE::BODY_STATE_CAN_SLEEP: {
			set_is_sleep_allowed(p_value);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled body state: '%d'. This should not happen. Please report this.", p_state));
		} break;
	}
}

Transform3D JoltSoftBody3D::get_transform() const {
	// Since any transform gets baked into the vertices anyway we can just return identity here.
	return Transform3D();
}

void JoltSoftBody3D::set_transform(const Transform3D &p_transform) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to set transform for '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	// For whatever reason this has to be interpreted as a relative global-space transform rather than an absolute one,
	// because `SoftBody3D` will immediately upon entering the scene tree set itself to be top-level and also set its
	// transform to be identity, while still expecting to stay in its original position.
	//
	// We also discard any scaling, since we have no way of scaling the actual edge lengths.
	const JPH::Mat44 relative_transform = to_jolt(p_transform.orthonormalized());

	// The translation delta goes to the body's position to avoid vertices getting too far away from it.
	JPH::BodyInterface &body_iface = space->get_body_iface();
	body_iface.SetPosition(jolt_body->GetID(), jolt_body->GetPosition() + relative_transform.GetTranslation(), JPH::EActivation::DontActivate);

	// The rotation difference goes to the vertices. We also reset the velocity of these vertices.
	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();

	for (JPH::SoftBodyVertex &vertex : physics_vertices) {
		vertex.mPosition = vertex.mPreviousPosition = relative_transform.Multiply3x3(vertex.mPosition);
		vertex.mVelocity = JPH::Vec3::sZero();
	}

	_transform_changed();
}

AABB JoltSoftBody3D::get_bounds() const {
	ERR_FAIL_COND_V_MSG(!in_space(), AABB(), vformat("Failed to retrieve world bounds of '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));
	return to_godot(jolt_body->GetWorldSpaceBounds());
}

void JoltSoftBody3D::update_rendering_server(PhysicsServer3DRenderingServerHandler *p_rendering_server_handler) {
	// Ideally we would emit an actual error here, but that would spam the logs to the point where the actual cause will be drowned out.
	if (unlikely(!in_space())) {
		return;
	}

	const JPH::SoftBodyMotionProperties &motion_properties = static_cast<const JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());

	typedef JPH::SoftBodyMotionProperties::Vertex SoftBodyVertex;
	typedef JPH::SoftBodyMotionProperties::Face SoftBodyFace;

	const JPH::Array<SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();
	const JPH::Array<SoftBodyFace> &physics_faces = motion_properties.GetFaces();

	const int physics_vertex_count = (int)physics_vertices.size();

	normals.clear();
	normals.resize(physics_vertex_count);

	// Compute vertex normals using smooth-shading:
	// Each vertex should use the average normal of all faces it is a part of.
	// Iterate over each face, and add the face normal to each of the face vertices.
	// By the end of the loop, each vertex normal will be the sum of all face normals it belongs to.
	for (const SoftBodyFace &physics_face : physics_faces) {
		// Jolt uses a different winding order, so we swap the indices to account for that.

		const uint32_t i0 = physics_face.mVertex[2];
		const uint32_t i1 = physics_face.mVertex[1];
		const uint32_t i2 = physics_face.mVertex[0];

		const Vector3 v0 = to_godot(physics_vertices[i0].mPosition);
		const Vector3 v1 = to_godot(physics_vertices[i1].mPosition);
		const Vector3 v2 = to_godot(physics_vertices[i2].mPosition);

		const Vector3 normal = (v2 - v0).cross(v1 - v0).normalized();

		normals[i0] += normal;
		normals[i1] += normal;
		normals[i2] += normal;
	}
	// Normalize the vertex normals to have length 1.0
	for (Vector3 &n : normals) {
		real_t len = n.length();
		// Some normals may have length 0 if the face was degenerate,
		// so don't divide by zero.
		if (len > CMP_EPSILON) {
			n /= len;
		}
	}

	const int mesh_vertex_count = mesh_to_physics.size();
	const JPH::RVec3 body_position = jolt_body->GetCenterOfMassPosition();

	for (int i = 0; i < mesh_vertex_count; ++i) {
		const int physics_index = mesh_to_physics[i];
		if (physics_index >= 0) {
			const Vector3 vertex = to_godot(body_position + physics_vertices[(size_t)physics_index].mPosition);
			const Vector3 normal = normals[(uint32_t)physics_index];

			p_rendering_server_handler->set_vertex(i, vertex);
			p_rendering_server_handler->set_normal(i, normal);
		}
	}

	p_rendering_server_handler->set_aabb(get_bounds());
}

Vector3 JoltSoftBody3D::get_vertex_position(int p_index) {
	ERR_FAIL_COND_V_MSG(!in_space(), Vector3(), vformat("Failed to retrieve point position for '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	ERR_FAIL_INDEX_V(p_index, (int)mesh_to_physics.size(), Vector3());
	const int physics_index = mesh_to_physics[p_index];
	ERR_FAIL_COND_V_MSG(physics_index < 0, Vector3(), vformat("Soft body vertex %d was not used by a face and has been omitted for '%s'. Position cannot be returned.", p_index, to_string()));

	const JPH::SoftBodyMotionProperties &motion_properties = static_cast<const JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	const JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();
	const JPH::SoftBodyVertex &physics_vertex = physics_vertices[physics_index];

	return to_godot(jolt_body->GetCenterOfMassPosition() + physics_vertex.mPosition);
}

void JoltSoftBody3D::set_vertex_position(int p_index, const Vector3 &p_position) {
	ERR_FAIL_COND_MSG(!in_space(), vformat("Failed to set point position for '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	ERR_FAIL_INDEX(p_index, (int)mesh_to_physics.size());
	const int physics_index = mesh_to_physics[p_index];
	ERR_FAIL_COND_MSG(physics_index < 0, vformat("Soft body vertex %d was not used by a face and has been omitted for '%s'. Position cannot be set.", p_index, to_string()));

	JPH::SoftBodyMotionProperties &motion_properties = static_cast<JPH::SoftBodyMotionProperties &>(*jolt_body->GetMotionPropertiesUnchecked());
	JPH::Array<JPH::SoftBodyVertex> &physics_vertices = motion_properties.GetVertices();
	JPH::SoftBodyVertex &physics_vertex = physics_vertices[physics_index];

	const JPH::RVec3 center_of_mass = jolt_body->GetCenterOfMassPosition();
	const JPH::Vec3 target_rel_pos = JPH::Vec3(to_jolt_r(p_position) - center_of_mass);

	const HashMap<int, int>::ConstIterator anchor_it = mesh_to_anchor.find(p_index);
	if (anchor_it) {
		const int anchor_index = anchor_it->value;
		if (anchor_index >= 0 && anchor_index < (int)physics_vertices.size()) {
			physics_vertices[anchor_index].mPosition = target_rel_pos;
			physics_vertices[anchor_index].mVelocity = JPH::Vec3::sZero();
		}
	} else {
		JPH::SoftBodyVertex &physics_vertex = physics_vertices[physics_index];
		physics_vertex.mPosition = target_rel_pos;
		physics_vertex.mVelocity = JPH::Vec3::sZero();
	}

	_vertices_changed();
}

void JoltSoftBody3D::pin_vertex(int p_index) {
	pinned_vertices.insert(p_index);

	_pins_changed();
}

void JoltSoftBody3D::unpin_vertex(int p_index) {
	pinned_vertices.erase(p_index);
	pinned_weights.erase(p_index);

	_pins_changed();
}

void JoltSoftBody3D::unpin_all_vertices() {
	pinned_vertices.clear();
	pinned_weights.clear();

	_pins_changed();
}

bool JoltSoftBody3D::is_vertex_pinned(int p_index) const {
	ERR_FAIL_COND_V_MSG(!in_space(), false, vformat("Failed retrieve pin status of point for '%s'. Doing so without a physics space is not supported when using Jolt Physics. If this relates to a node, try adding the node to a scene tree first.", to_string()));

	ERR_FAIL_INDEX_V(p_index, (int)mesh_to_physics.size(), false);

	return pinned_vertices.has(p_index);
}

void JoltSoftBody3D::set_vertex_weight(int p_index, float p_weight) {
	pinned_weights[p_index] = CLAMP(p_weight, 0.0f, 1.0f);
	_pins_changed();
}

float JoltSoftBody3D::get_vertex_weight(int p_index) const {
	const HashMap<int, float>::ConstIterator E = pinned_weights.find(p_index);
	if (E) {
		return E->value;
	}
	return 1.0f;
}
