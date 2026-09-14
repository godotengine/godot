/**************************************************************************/
/*  soft_body_cap_probe.h                                                 */
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

#pragma once

#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"

// Keep tests Jolt-header-free: missing JPH_DEBUG_RENDERER causes layout mismatch.
// Exchange POD values with the .cpp compiled by env_jolt.
namespace JoltSoftBodyCapProbe {

// True when the RID resolves to a Jolt soft body that is in a space.
bool in_space(RID p_body);

int vertex_count(RID p_body);
float vertex_inv_mass(RID p_body, int p_index);
Vector3 vertex_position(RID p_body, int p_index);
Vector3 center_of_mass(RID p_body);

int face_count(RID p_body);
int edge_count(RID p_body);

// Constraint counts for baseline-gap checks and capability coverage.
int dihedral_bend_count(RID p_body);
int lra_count(RID p_body);
int volume_count(RID p_body);
int skinned_count(RID p_body);

// Invalid body/index produces valid=false, never an unchecked Jolt array read.
struct Constraint {
	bool valid = false;
	int vertices[4] = { -1, -1, -1, -1 };
	float compliance = 0.0f;
	float rest = 0.0f;
};
struct SkinConstraint {
	bool valid = false;
	int vertex = -1;
	int joints[4] = { -1, -1, -1, -1 };
	float weights[4] = {};
	uint32_t normal_info = 0;
	float max_distance = 0, back_stop_distance = 0, back_stop_radius = 0;
};
struct BuildCounts {
	uint64_t create_constraints = 0;
	uint64_t optimize = 0;
	uint64_t generation = 0;
	uint64_t pre_step = 0;
	uint64_t skin_init = 0;
	uint64_t skin_recurring = 0;
	uint64_t renderer_reads = 0;
	uint64_t body_creations = 0;
	uint64_t solver_steps = 0;
	uint64_t removals_during_traversal = 0;
};
Constraint edge(RID p_body, int p_index);
Constraint face(RID p_body, int p_index);
int update_group_count(RID p_body);
Constraint dihedral(RID p_body, int p_index);
Constraint lra(RID p_body, int p_index);
Constraint tetra(RID p_body, int p_index);
SkinConstraint skin(RID p_body, int p_index);
int inv_bind_count(RID p_body);
float shared_inv_mass(RID p_body, int p_index);
Vector3 world_position(RID p_body, int p_index);
Vector3 velocity(RID p_body, int p_index);
Transform3D frame(RID p_body);
uint64_t body_identity(RID p_body);
uint64_t settings_identity(RID p_body);
bool all_finite(RID p_body);
BuildCounts build_counts(RID p_body);
struct SkinTarget {
	bool valid = false;
	Vector3 previous;
	Vector3 current;
	Vector3 normal;
};
struct Scalars {
	bool valid = false;
	float friction = 0;
	float restitution = 0;
	float gravity_factor = 0;
	float vertex_radius = 0;
	bool faces_double_sided = false;
	float pressure = 0, damping = 0;
	uint32_t iterations = 0;
};
SkinTarget skin_target(RID p_body, int p_vertex);
Scalars scalars(RID p_body);
class ContactScope {
	void *implementation = nullptr;

public:
	explicit ContactScope(RID p_body);
	~ContactScope();
	ContactScope(const ContactScope &) = delete;
	ContactScope &operator=(const ContactScope &) = delete;
	int count() const;
};
bool face_ray_hit(RID p_body, const Vector3 &p_origin, const Vector3 &p_direction);
bool skip_next_skin_call(RID p_body);
bool volume_control(RID p_body);
bool clear_faces(RID p_body);
bool set_body_frame(RID p_body, const Transform3D &p_frame);
bool set_vertex_local(RID p_body, int p_vertex, const Vector3 &p_position, const Vector3 &p_velocity = Vector3());
bool invoke_pre_step(RID p_body, float p_step = 1.0f / 60.0f);
// Direct Jolt fixture: None=0, Distance=1, Dihedral=2. No registry emulation.
Constraint fixture_bend(bool p_square, int p_mode, int &r_edges, int &r_dihedrals);

int rod_count(RID p_body);
int rod_bend_twist_count(RID p_body);
// mRodStates has no size accessor; read its last constraint-indexed entry.
bool rod_state_is_readable(RID p_body, int p_index);
float rod_length(RID p_body, int p_index);
float min_rod_length(RID p_body);
bool all_bishop_frames_set(RID p_body);
bool rod_angular_velocities_are_zero(RID p_body);
// `p_which` is 0 or 1.
int rod_vertex(RID p_body, int p_rod, int p_which);
int bend_twist_rod(RID p_body, int p_bend, int p_which);
float rod_compliance(RID p_body, int p_index);
float bend_twist_compliance(RID p_body, int p_index);

bool update_position(RID p_body);

// Detect the backend itself; PhysicsServer3DManager returns a wrapper, hence false.
bool is_inner_jolt_server(const void *p_server);

// Expose the registry without including its Jolt-dependent header.
int capability_count();
String capability_prefix(int p_cap);
String capability_required_key(int p_cap);
int capability_property_count(int p_cap);
String capability_property_name(int p_cap, int p_prop);
// Return Variant::Type as int to keep engine enums out of the interface.
int capability_property_type(int p_cap, int p_prop);
bool capability_property_live_read(int p_cap, int p_prop);
bool capability_property_stored(int p_cap, int p_prop);
bool capability_property_clearable(int p_cap, int p_prop);
bool capability_property_rebuild(int p_cap, int p_prop);

// The build-time list of `capabilities/cap_*.cpp`, basenames only.
int capability_source_file_count();
String capability_source_file(int p_index);

// Own a real threaded wrapper/backend pair without exposing JPH types.
class ThreadedServerScope {
	void *implementation = nullptr;

public:
	ThreadedServerScope();
	~ThreadedServerScope();
	void *server() const;
	bool actual_separate_thread() const;
};

// Fresh-settings checks requiring no body, space, server or RID.
bool calculate_rod_properties_keeps_chain_order(int p_joints);
bool optimize_is_identity_for_chain(int p_joints);

} // namespace JoltSoftBodyCapProbe
