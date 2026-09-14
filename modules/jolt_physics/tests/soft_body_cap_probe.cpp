/**************************************************************************/
/*  soft_body_cap_probe.cpp                                               */
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

#ifdef TESTS_ENABLED

#include "soft_body_cap_probe.h"

#include "../capabilities/soft_body_capabilities.h"
#include "../jolt_physics_server_3d.h"
#include "../misc/jolt_type_conversions.h"
#include "../objects/jolt_soft_body_3d.h"
#include "../spaces/jolt_job_system.h"
#include "../spaces/jolt_space_3d.h"

#include "core/math/math_funcs.h"
#include "core/object/worker_thread_pool.h"
#include "core/os/semaphore.h"
#include "servers/physics_3d/physics_server_3d_wrap_mt.h"

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeFilter.h>
#include <Jolt/Physics/SoftBody/SoftBodyContactListener.h>
#include <Jolt/Physics/SoftBody/SoftBodyManifold.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/Physics/StateRecorder.h>

#include <atomic>
#include <cfloat>

namespace {

const JPH::SoftBodyMotionProperties *resolve(RID p_body) {
	JoltPhysicsServer3D *server = JoltPhysicsServer3D::get_singleton();
	if (server == nullptr) {
		return nullptr;
	}
	JoltSoftBody3D *body = server->get_soft_body_for_tests(p_body);
	if (body == nullptr || !body->in_space()) {
		return nullptr;
	}
	return static_cast<const JPH::SoftBodyMotionProperties *>(body->get_jolt_body()->GetMotionPropertiesUnchecked());
}

// Read private SkinState through SaveState; derive offsets from the base serializer.
class SnapshotRecorder final : public JPH::StateRecorder {
public:
	Vector<uint8_t> bytes;
	bool failed = false;
	void WriteBytes(const void *p_data, size_t p_size) override {
		const int offset = bytes.size();
		bytes.resize(offset + p_size);
		memcpy(bytes.ptrw() + offset, p_data, p_size);
	}
	void ReadBytes(void *, size_t) override { failed = true; }
	bool IsEOF() const override { return failed; }
	bool IsFailed() const override { return failed; }
};

// Unit-spaced +X chain: N-1 stretch-shear rods and N-2 bend-twist constraints.
void build_chain(JPH::SoftBodySharedSettings &r_settings, int p_joints) {
	for (int i = 0; i < p_joints; i++) {
		JPH::SoftBodySharedSettings::Vertex vertex;
		vertex.mPosition = JPH::Float3((float)i, 0.0f, 0.0f);
		vertex.mInvMass = 1.0f;
		r_settings.mVertices.push_back(vertex);
	}
	for (int i = 0; i + 1 < p_joints; i++) {
		r_settings.mRodStretchShearConstraints.push_back(JPH::SoftBodySharedSettings::RodStretchShear((JPH::uint32)i, (JPH::uint32)(i + 1), 0.0f));
	}
	for (int i = 0; i + 2 < p_joints; i++) {
		r_settings.mRodBendTwistConstraints.push_back(JPH::SoftBodySharedSettings::RodBendTwist((JPH::uint32)i, (JPH::uint32)(i + 1), 0.0f));
	}
}

} // namespace

namespace JoltSoftBodyCapProbe {

bool in_space(RID p_body) {
	return resolve(p_body) != nullptr;
}

int vertex_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetVertices().size();
}

float vertex_inv_mass(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetVertices().size()) {
		return -1.0f;
	}
	return motion->GetVertex((JPH::uint)p_index).mInvMass;
}

Vector3 vertex_position(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetVertices().size()) {
		return Vector3();
	}
	const JPH::Vec3 position = motion->GetVertex((JPH::uint)p_index).mPosition;
	return Vector3(position.GetX(), position.GetY(), position.GetZ());
}

Vector3 center_of_mass(RID p_body) {
	JoltPhysicsServer3D *server = JoltPhysicsServer3D::get_singleton();
	if (server == nullptr) {
		return Vector3();
	}
	JoltSoftBody3D *body = server->get_soft_body_for_tests(p_body);
	if (body == nullptr || !body->in_space()) {
		return Vector3();
	}
	const JPH::RVec3 position = body->get_jolt_body()->GetCenterOfMassPosition();
	return Vector3((real_t)position.GetX(), (real_t)position.GetY(), (real_t)position.GetZ());
}

int face_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mFaces.size();
}

int edge_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mEdgeConstraints.size();
}

int dihedral_bend_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mDihedralBendConstraints.size();
}

int lra_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mLRAConstraints.size();
}

int volume_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mVolumeConstraints.size();
}

int skinned_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mSkinnedConstraints.size();
}

Constraint face(RID p_body, int p_index) {
	Constraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= int(motion->GetSettings()->mFaces.size())) {
		return result;
	}
	result.valid = true;
	for (int i = 0; i < 3; ++i) {
		result.vertices[i] = motion->GetSettings()->mFaces[p_index].mVertex[i];
	}
	return result;
}
int update_group_count(RID p_body) {
	const auto *motion = resolve(p_body);
	if (motion == nullptr) {
		return -1;
	}
	const auto &settings = *motion->GetSettings();
	// Read private update groups through the public binary-state stream.
	SnapshotRecorder prefix, full;
	prefix.Write(settings.mVertices);
	prefix.Write(settings.mFaces);
	prefix.Write(settings.mEdgeConstraints);
	prefix.Write(settings.mDihedralBendConstraints);
	prefix.Write(settings.mVolumeConstraints);
	prefix.Write(settings.mSkinnedConstraints);
	settings.SaveBinaryState(full);
	size_t offset = prefix.bytes.size();
	for (size_t stride : { sizeof(uint32_t), sizeof(JPH::SoftBodySharedSettings::LRA) }) {
		if (offset + sizeof(uint32_t) > size_t(full.bytes.size())) {
			return -1;
		}
		uint32_t count;
		memcpy(&count, full.bytes.ptr() + offset, sizeof(count));
		offset += sizeof(count);
		if (count > (size_t(full.bytes.size()) - offset) / stride) {
			return -1;
		}
		offset += count * stride;
	}
	if (offset + sizeof(uint32_t) > size_t(full.bytes.size())) {
		return -1;
	}
	uint32_t count;
	memcpy(&count, full.bytes.ptr() + offset, sizeof(count));
	return count <= INT32_MAX ? int(count) : -1;
}

Constraint edge(RID p_body, int p_index) {
	Constraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mEdgeConstraints.size()) {
		return result;
	}
	const auto &constraint = motion->GetSettings()->mEdgeConstraints[p_index];
	result.valid = true;
	for (int i = 0; i < 2; ++i) {
		result.vertices[i] = constraint.mVertex[i];
	}
	result.rest = constraint.mRestLength;
	result.compliance = constraint.mCompliance;
	return result;
}

Constraint dihedral(RID p_body, int p_index) {
	Constraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mDihedralBendConstraints.size()) {
		return result;
	}
	const auto &constraint = motion->GetSettings()->mDihedralBendConstraints[p_index];
	result.valid = true;
	for (int i = 0; i < 4; ++i) {
		result.vertices[i] = constraint.mVertex[i];
	}
	result.rest = constraint.mInitialAngle;
	result.compliance = constraint.mCompliance;
	return result;
}

Constraint lra(RID p_body, int p_index) {
	Constraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mLRAConstraints.size()) {
		return result;
	}
	const auto &constraint = motion->GetSettings()->mLRAConstraints[p_index];
	result.valid = true;
	for (int i = 0; i < 2; ++i) {
		result.vertices[i] = constraint.mVertex[i];
	}
	result.rest = constraint.mMaxDistance;
	return result;
}

Constraint tetra(RID p_body, int p_index) {
	Constraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mVolumeConstraints.size()) {
		return result;
	}
	const auto &constraint = motion->GetSettings()->mVolumeConstraints[p_index];
	result.valid = true;
	for (int i = 0; i < 4; ++i) {
		result.vertices[i] = constraint.mVertex[i];
	}
	result.rest = constraint.mSixRestVolume;
	result.compliance = constraint.mCompliance;
	return result;
}

SkinConstraint skin(RID p_body, int p_index) {
	SkinConstraint result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mSkinnedConstraints.size()) {
		return result;
	}
	const auto &constraint = motion->GetSettings()->mSkinnedConstraints[p_index];
	result.valid = true;
	result.vertex = constraint.mVertex;
	result.normal_info = constraint.mNormalInfo;
	result.max_distance = constraint.mMaxDistance;
	result.back_stop_distance = constraint.mBackStopDistance;
	result.back_stop_radius = constraint.mBackStopRadius;
	for (int i = 0; i < 4; ++i) {
		result.joints[i] = constraint.mWeights[i].mInvBindIndex;
		result.weights[i] = constraint.mWeights[i].mWeight;
	}
	return result;
}

int inv_bind_count(RID p_body) {
	const auto *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mInvBindMatrices.size();
}

float shared_inv_mass(RID p_body, int p_index) {
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mVertices.size()) {
		return -1.0f;
	}
	return motion->GetSettings()->mVertices[p_index].mInvMass;
}

Vector3 velocity(RID p_body, int p_index) {
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetVertices().size()) {
		return Vector3();
	}
	return to_godot(motion->GetVertex(p_index).mVelocity);
}

Transform3D frame(RID p_body) {
	if (resolve(p_body) == nullptr) {
		return Transform3D();
	}
	return to_godot(JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body)->get_jolt_body()->GetCenterOfMassTransform());
}

Vector3 world_position(RID p_body, int p_index) {
	if (resolve(p_body) == nullptr || p_index < 0 || p_index >= vertex_count(p_body)) {
		return Vector3();
	}
	return frame(p_body).xform(vertex_position(p_body, p_index));
}

uint64_t body_identity(RID p_body) {
	if (resolve(p_body) == nullptr) {
		return UINT64_MAX;
	}
	return JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body)->get_jolt_body()->GetID().GetIndexAndSequenceNumber();
}

uint64_t settings_identity(RID p_body) {
	const auto *motion = resolve(p_body);
	return motion == nullptr ? 0 : reinterpret_cast<uintptr_t>(motion->GetSettings());
}

bool all_finite(RID p_body) {
	const auto *motion = resolve(p_body);
	if (motion == nullptr || !frame(p_body).is_finite()) {
		return false;
	}
	for (const auto &vertex : motion->GetVertices()) {
		if (!to_godot(vertex.mPosition).is_finite() || !to_godot(vertex.mVelocity).is_finite() || !Math::is_finite(vertex.mInvMass)) {
			return false;
		}
	}
	return true;
}

BuildCounts build_counts(RID p_body) {
	auto *server = JoltPhysicsServer3D::get_singleton();
	auto *body = server == nullptr ? nullptr : server->get_soft_body_for_tests(p_body);
	return body == nullptr ? BuildCounts() : body->get_cap_build_counts_for_tests();
}

SkinTarget skin_target(RID p_body, int p_vertex) {
	SkinTarget result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr || p_vertex < 0 || p_vertex >= (int)motion->GetVertices().size()) {
		return result;
	}
	bool selected = false;
	for (const auto &constraint : motion->GetSettings()->mSkinnedConstraints) {
		selected |= constraint.mVertex == (uint32_t)p_vertex;
	}
	if (!selected) {
		return result;
	}
	SnapshotRecorder base;
	motion->JPH::MotionProperties::SaveState(base);
	SnapshotRecorder full;
	motion->SaveState(full);
	const size_t vector_size = 3 * sizeof(float);
	const size_t offset = base.bytes.size() + motion->GetVertices().size() * 2 * vector_size + motion->GetSettings()->mRodStretchShearConstraints.size() * (sizeof(JPH::Quat) + vector_size) + p_vertex * 3 * vector_size;
	if (offset + 3 * vector_size > (size_t)full.bytes.size()) {
		return result;
	}
	float values[9];
	memcpy(values, full.bytes.ptr() + offset, sizeof(values));
	result.valid = true;
	result.previous = Vector3(values[0], values[1], values[2]);
	result.current = Vector3(values[3], values[4], values[5]);
	result.normal = Vector3(values[6], values[7], values[8]);
	return result;
}

Scalars scalars(RID p_body) {
	Scalars result;
	const auto *motion = resolve(p_body);
	if (motion == nullptr) {
		return result;
	}
	const auto *body = JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body)->get_jolt_body();
	result.valid = true;
	result.friction = body->GetFriction();
	result.restitution = body->GetRestitution();
	result.gravity_factor = motion->GetGravityFactor();
	result.vertex_radius = motion->GetVertexRadius();
	result.faces_double_sided = motion->GetFacesDoubleSided();
	result.pressure = motion->GetPressure();
	result.damping = motion->GetLinearDamping();
	result.iterations = motion->GetNumIterations();
	return result;
}

namespace {
class ContactRecorder final : public JPH::SoftBodyContactListener {
public:
	JPH::PhysicsSystem &system;
	JPH::SoftBodyContactListener *previous;
	JPH::BodyID target;
	std::atomic<int> contacts{ 0 };
	ContactRecorder(JPH::PhysicsSystem &p_system, JPH::BodyID p_target) :
			system(p_system), previous(system.GetSoftBodyContactListener()), target(p_target) {
		system.SetSoftBodyContactListener(this);
	}
	~ContactRecorder() override { system.SetSoftBodyContactListener(previous); }
	JPH::SoftBodyValidateResult OnSoftBodyContactValidate(const JPH::Body &p_body, const JPH::Body &p_other, JPH::SoftBodyContactSettings &p_settings) override {
		return previous == nullptr ? JPH::SoftBodyValidateResult::AcceptContact : previous->OnSoftBodyContactValidate(p_body, p_other, p_settings);
	}
	void OnSoftBodyContactAdded(const JPH::Body &p_body, const JPH::SoftBodyManifold &p_manifold) override {
		if (p_body.GetID() == target) {
			for (const auto &vertex : p_manifold.GetVertices()) {
				if (p_manifold.HasContact(vertex)) {
					contacts.fetch_add(1, std::memory_order_relaxed);
				}
			}
		}
		if (previous != nullptr) {
			previous->OnSoftBodyContactAdded(p_body, p_manifold);
		}
	}
};
} // namespace

ContactScope::ContactScope(RID p_body) {
	if (resolve(p_body) != nullptr) {
		auto *body = JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body);
		implementation = memnew(ContactRecorder(body->get_space()->get_physics_system(), body->get_jolt_body()->GetID()));
	}
}
ContactScope::~ContactScope() {
	if (implementation != nullptr) {
		memdelete(static_cast<ContactRecorder *>(implementation));
	}
}
int ContactScope::count() const {
	return implementation == nullptr ? -1 : static_cast<ContactRecorder *>(implementation)->contacts.load(std::memory_order_relaxed);
}

bool face_ray_hit(RID p_body, const Vector3 &p_origin, const Vector3 &p_direction) {
	if (resolve(p_body) == nullptr) {
		return false;
	}
	const auto *body = JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body)->get_jolt_body();
	JPH::RayCastSettings settings;
	settings.mBackFaceModeTriangles = JPH::EBackFaceMode::IgnoreBackFaces;
	JPH::AllHitCollisionCollector<JPH::CastRayCollector> collector;
	body->GetShape()->CastRay(JPH::RayCast(to_jolt(p_origin), to_jolt(p_direction)), settings, JPH::SubShapeIDCreator(), collector, JPH::ShapeFilter());
	return collector.HadHit();
}

bool volume_control(RID p_body) {
	auto *server = JoltPhysicsServer3D::get_singleton();
	auto *body = server == nullptr ? nullptr : server->get_soft_body_for_tests(p_body);
	if (body == nullptr || body->in_space()) {
		return false;
	}
	body->set_volume_control_for_tests();
	return true;
}

bool set_body_frame(RID p_body, const Transform3D &p_frame) {
	if (resolve(p_body) == nullptr || !p_frame.is_finite()) {
		return false;
	}
	auto *body = JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body);
	body->get_space()->get_body_iface().SetPositionAndRotation(body->get_jolt_body()->GetID(), to_jolt_r(p_frame.origin), to_jolt(p_frame.basis), JPH::EActivation::Activate);
	return true;
}

bool clear_faces(RID p_body) {
	const auto *motion = resolve(p_body);
	if (motion == nullptr) {
		return false;
	}
	const_cast<JPH::SoftBodySharedSettings *>(motion->GetSettings())->mFaces.clear();
	return true;
}

bool set_vertex_local(RID p_body, int p_vertex, const Vector3 &p_position, const Vector3 &p_velocity) {
	auto *motion = const_cast<JPH::SoftBodyMotionProperties *>(resolve(p_body));
	if (motion == nullptr || p_vertex < 0 || p_vertex >= int(motion->GetVertices().size()) || !p_position.is_finite() || !p_velocity.is_finite()) {
		return false;
	}
	auto &vertex = motion->GetVertex(p_vertex);
	vertex.mPosition = vertex.mPreviousPosition = to_jolt(p_position);
	vertex.mVelocity = to_jolt(p_velocity);
	return true;
}

bool skip_next_skin_call(RID p_body) {
	auto *server = JoltPhysicsServer3D::get_singleton();
	auto *body = server == nullptr ? nullptr : server->get_soft_body_for_tests(p_body);
	if (body == nullptr) {
		return false;
	}
	body->skip_next_skin_call_for_tests();
	return true;
}

bool invoke_pre_step(RID p_body, float p_step) {
	if (resolve(p_body) == nullptr || !Math::is_finite(p_step) || p_step <= 0) {
		return false;
	}
	JoltPhysicsServer3D::get_singleton()->get_soft_body_for_tests(p_body)->pre_step(p_step);
	return true;
}

Constraint fixture_bend(bool p_square, int p_mode, int &r_edges, int &r_dihedrals) {
	JPH::SoftBodySharedSettings settings;
	settings.mVertices.emplace_back(JPH::Float3(0, 0, 0));
	settings.mVertices.emplace_back(JPH::Float3(1, 0, 0));
	settings.mVertices.emplace_back(JPH::Float3(1, 0, 1));
	settings.mVertices.emplace_back(JPH::Float3(p_square ? 0.0f : -0.7f, 0, p_square ? 1.0f : 0.4f));
	settings.mFaces.emplace_back(2, 1, 0);
	settings.mFaces.emplace_back(3, 2, 0);
	JPH::SoftBodySharedSettings::VertexAttributes attributes;
	attributes.mBendCompliance = 1.0e-4f;
	const auto mode = p_mode == 1 ? JPH::SoftBodySharedSettings::EBendType::Distance : p_mode == 2 ? JPH::SoftBodySharedSettings::EBendType::Dihedral
																								   : JPH::SoftBodySharedSettings::EBendType::None;
	settings.CreateConstraints(&attributes, 1, mode);
	r_edges = settings.mEdgeConstraints.size();
	r_dihedrals = settings.mDihedralBendConstraints.size();
	Constraint result;
	for (const auto &edge : settings.mEdgeConstraints) {
		if ((edge.mVertex[0] == 1 && edge.mVertex[1] == 3) || (edge.mVertex[0] == 3 && edge.mVertex[1] == 1)) {
			result.valid = true;
			result.vertices[0] = edge.mVertex[0];
			result.vertices[1] = edge.mVertex[1];
			result.rest = edge.mRestLength;
			result.compliance = edge.mCompliance;
		}
	}
	return result;
}

int rod_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mRodStretchShearConstraints.size();
}

int rod_bend_twist_count(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? -1 : (int)motion->GetSettings()->mRodBendTwistConstraints.size();
}

bool rod_state_is_readable(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mRodStretchShearConstraints.size()) {
		return false;
	}
	// Initialize sizes mRodStates by constraint count; undersizing makes this
	// read out of bounds.
	return motion->GetRodRotation((JPH::uint)p_index).IsNormalized();
}

float rod_length(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mRodStretchShearConstraints.size()) {
		return -1.0f;
	}
	return motion->GetSettings()->mRodStretchShearConstraints[p_index].mLength;
}

float min_rod_length(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr) {
		return -1.0f;
	}
	float smallest = FLT_MAX;
	for (const JPH::SoftBodySharedSettings::RodStretchShear &rod : motion->GetSettings()->mRodStretchShearConstraints) {
		smallest = MIN(smallest, rod.mLength);
	}
	return smallest == FLT_MAX ? -1.0f : smallest;
}

bool all_bishop_frames_set(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr) {
		return false;
	}
	const JPH::Array<JPH::SoftBodySharedSettings::RodStretchShear> &rods = motion->GetSettings()->mRodStretchShearConstraints;
	if (rods.empty()) {
		return false;
	}
	for (const JPH::SoftBodySharedSettings::RodStretchShear &rod : rods) {
		// Missing CalculateRodProperties leaves zero; zero-length segments produce NaN.
		if (rod.mBishop == JPH::Quat::sZero() || !rod.mBishop.IsNormalized()) {
			return false;
		}
	}
	return true;
}

bool rod_angular_velocities_are_zero(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr) {
		return false;
	}
	const int count = (int)motion->GetSettings()->mRodStretchShearConstraints.size();
	for (int i = 0; i < count; i++) {
		if (!motion->GetRodAngularVelocity((JPH::uint)i).IsNearZero()) {
			return false;
		}
	}
	return true;
}

int rod_vertex(RID p_body, int p_rod, int p_which) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_which < 0 || p_which > 1) {
		return -1;
	}
	const JPH::Array<JPH::SoftBodySharedSettings::RodStretchShear> &rods = motion->GetSettings()->mRodStretchShearConstraints;
	if (p_rod < 0 || p_rod >= (int)rods.size()) {
		return -1;
	}
	return (int)rods[p_rod].mVertex[p_which];
}

int bend_twist_rod(RID p_body, int p_bend, int p_which) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_which < 0 || p_which > 1) {
		return -1;
	}
	const JPH::Array<JPH::SoftBodySharedSettings::RodBendTwist> &bends = motion->GetSettings()->mRodBendTwistConstraints;
	if (p_bend < 0 || p_bend >= (int)bends.size()) {
		return -1;
	}
	return (int)bends[p_bend].mRod[p_which];
}

float rod_compliance(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mRodStretchShearConstraints.size()) {
		return -1.0f;
	}
	return motion->GetSettings()->mRodStretchShearConstraints[p_index].mCompliance;
}

float bend_twist_compliance(RID p_body, int p_index) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	if (motion == nullptr || p_index < 0 || p_index >= (int)motion->GetSettings()->mRodBendTwistConstraints.size()) {
		return -1.0f;
	}
	return motion->GetSettings()->mRodBendTwistConstraints[p_index].mCompliance;
}

bool update_position(RID p_body) {
	const JPH::SoftBodyMotionProperties *motion = resolve(p_body);
	return motion == nullptr ? false : motion->GetUpdatePosition();
}

bool is_inner_jolt_server(const void *p_server) {
	return (const void *)JoltPhysicsServer3D::get_singleton() == p_server;
}

int capability_count() {
	return (int)all_capabilities().size();
}

String capability_prefix(int p_cap) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), String());
	return String(all_capabilities()[p_cap]->prefix);
}

String capability_required_key(int p_cap) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), String());
	return String(all_capabilities()[p_cap]->required_key);
}

int capability_property_count(int p_cap) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), -1);
	return (int)all_capabilities()[p_cap]->props.size();
}

String capability_property_name(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), String());
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), String());
	return String(all_capabilities()[p_cap]->props[p_prop].name);
}

int capability_property_type(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), -1);
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), -1);
	return (int)all_capabilities()[p_cap]->props[p_prop].type;
}

bool capability_property_live_read(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), false);
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), false);
	return all_capabilities()[p_cap]->props[p_prop].live_read;
}

bool capability_property_stored(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), false);
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), false);
	return all_capabilities()[p_cap]->props[p_prop].stored;
}
bool capability_property_clearable(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), false);
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), false);
	return all_capabilities()[p_cap]->props[p_prop].clearable;
}
bool capability_property_rebuild(int p_cap, int p_prop) {
	ERR_FAIL_INDEX_V(p_cap, capability_count(), false);
	ERR_FAIL_INDEX_V(p_prop, capability_property_count(p_cap), false);
	return all_capabilities()[p_cap]->props[p_prop].rebuild_on_write;
}

namespace {
// Drain jobs through their owning allocator before switching test servers.
// Update's barrier can precede trailing Release() calls; occupy each worker
// once to fence those callbacks without sleeps or solver changes.
void finish_worker_callbacks() {
	struct Fence {
		Semaphore entered;
		Semaphore release;
		static void execute(void *p_data) {
			auto *fence = static_cast<Fence *>(p_data);
			fence->entered.post();
			fence->release.wait();
		}
	} fence;
	auto *pool = WorkerThreadPool::get_singleton();
	LocalVector<WorkerThreadPool::TaskID> tasks;
	for (int i = 0; i < pool->get_thread_count(); ++i) {
		tasks.push_back(pool->add_native_task(&Fence::execute, &fence, true, "Soft-body test backend handoff"));
	}
	for (uint32_t i = 0; i < tasks.size(); ++i) {
		fence.entered.wait();
	}
	for (uint32_t i = 0; i < tasks.size(); ++i) {
		fence.release.post();
	}
	for (auto task : tasks) {
		pool->wait_for_task_completion(task);
	}
}

struct ThreadedBackend {
	PhysicsServer3D *saved = PhysicsServer3D::get_singleton();
	JoltPhysicsServer3D *saved_jolt = JoltPhysicsServer3D::get_singleton();
	JoltPhysicsServer3D *inner = memnew(JoltPhysicsServer3D(true));
	PhysicsServer3DWrapMT *wrapper = memnew(PhysicsServer3DWrapMT(inner, true));
	ThreadedBackend() {
		finish_worker_callbacks();
		if (saved_jolt != nullptr && saved_jolt->get_job_system_for_tests() != nullptr) {
			saved_jolt->get_job_system_for_tests()->post_step();
		}
		wrapper->init();
		wrapper->set_active(true);
	}
	~ThreadedBackend() {
		wrapper->sync();
		wrapper->end_sync();
		// Queued tasks may not wake an idle pump: join it before fencing workers.
		// Keep its allocator until all callbacks return, then drain its jobs.
		JoltJobSystem *jobs = inner->release_job_system_for_tests();
		wrapper->finish();
		finish_worker_callbacks();
		delete jobs;
		memdelete(wrapper);
		JoltPhysicsServer3D::set_singleton_for_tests(saved_jolt);
		PhysicsServer3D::set_singleton_for_tests(saved);
	}
};
} //namespace
ThreadedServerScope::ThreadedServerScope() {
	implementation = memnew(ThreadedBackend);
}
ThreadedServerScope::~ThreadedServerScope() {
	memdelete(static_cast<ThreadedBackend *>(implementation));
}
void *ThreadedServerScope::server() const {
	return static_cast<ThreadedBackend *>(implementation)->wrapper;
}
bool ThreadedServerScope::actual_separate_thread() const {
	return static_cast<ThreadedBackend *>(implementation)->inner->is_on_separate_thread();
}

int capability_source_file_count() {
	return (int)(sizeof(CAP_SOURCE_FILES) / sizeof(CAP_SOURCE_FILES[0]));
}

String capability_source_file(int p_index) {
	ERR_FAIL_INDEX_V(p_index, capability_source_file_count(), String());
	return String(CAP_SOURCE_FILES[p_index]);
}

bool calculate_rod_properties_keeps_chain_order(int p_joints) {
	if (p_joints < 2) {
		return false;
	}
	JPH::SoftBodySharedSettings settings;
	build_chain(settings, p_joints);
	settings.CalculateRodProperties();

	const JPH::Array<JPH::SoftBodySharedSettings::RodStretchShear> &rods = settings.mRodStretchShearConstraints;
	for (int i = 0; i < (int)rods.size(); i++) {
		if (rods[i].mVertex[0] != (JPH::uint32)i || rods[i].mVertex[1] != (JPH::uint32)(i + 1)) {
			return false;
		}
		// Every segment of the chain is one unit long, laid out along +X.
		if (!Math::is_equal_approx((real_t)rods[i].mLength, (real_t)1.0)) {
			return false;
		}
		if (rods[i].mBishop == JPH::Quat::sZero() || !rods[i].mBishop.IsNormalized()) {
			return false;
		}
		// The Bishop frame's Z axis is the segment tangent
		// (`SoftBodySharedSettings.cpp`: the tangent is the third column).
		const JPH::Vec3 tangent = rods[i].mBishop * JPH::Vec3(0, 0, 1);
		if (!Math::is_equal_approx((real_t)tangent.GetX(), (real_t)1.0, (real_t)1e-4)) {
			return false;
		}
	}
	const JPH::Array<JPH::SoftBodySharedSettings::RodBendTwist> &bends = settings.mRodBendTwistConstraints;
	for (int i = 0; i < (int)bends.size(); i++) {
		if (bends[i].mRod[0] != (JPH::uint32)i || bends[i].mRod[1] != (JPH::uint32)(i + 1)) {
			return false;
		}
		// A straight chain has no twist between adjacent rods.
		if (!bends[i].mOmega0.IsNormalized()) {
			return false;
		}
	}
	return true;
}

bool optimize_is_identity_for_chain(int p_joints) {
	if (p_joints < 2) {
		return false;
	}
	JPH::SoftBodySharedSettings settings;
	build_chain(settings, p_joints);
	settings.CalculateRodProperties();

	JPH::SoftBodySharedSettings::OptimizationResults results;
	settings.Optimize(results);

	// Identity remapping preserves authored indices: rod/state[i] is mRodStates[i].
	if (results.mRodStretchShearConstraintRemap.size() != (size_t)(p_joints - 1)) {
		return false;
	}
	for (JPH::uint i = 0; i < results.mRodStretchShearConstraintRemap.size(); i++) {
		if (results.mRodStretchShearConstraintRemap[i] != i) {
			return false;
		}
	}

	// Bend-twist order is private and may change. Require a permutation that
	// preserves adjacent rod pairs.
	const size_t bend_count = (size_t)MAX(p_joints - 2, 0);
	if (results.mRodBendTwistConstraintRemap.size() != bend_count) {
		return false;
	}
	JPH::Array<bool> seen;
	seen.resize(bend_count, false);
	for (JPH::uint i = 0; i < results.mRodBendTwistConstraintRemap.size(); i++) {
		const JPH::uint mapped = results.mRodBendTwistConstraintRemap[i];
		if (mapped >= (JPH::uint)bend_count || seen[mapped]) {
			return false;
		}
		seen[mapped] = true;
	}
	for (const JPH::SoftBodySharedSettings::RodBendTwist &bend : settings.mRodBendTwistConstraints) {
		if (bend.mRod[1] != bend.mRod[0] + 1) {
			return false;
		}
	}
	return true;
}

} // namespace JoltSoftBodyCapProbe

#endif // TESTS_ENABLED
