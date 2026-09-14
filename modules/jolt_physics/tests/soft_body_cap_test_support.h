/**************************************************************************/
/*  soft_body_cap_test_support.h                                          */
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

#include "soft_body_cap_probe.h"

#include "core/config/project_settings.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/variant/typed_array.h"
#include "servers/physics_3d/physics_server_3d.h"
#include "servers/physics_3d/physics_server_3d_dummy.h"
#include "servers/physics_3d/physics_server_3d_extension.h"
#include "servers/physics_3d/physics_server_3d_manager.h"
#include "servers/rendering/rendering_server.h"
#include "tests/test_macros.h"

namespace TestJoltSoftBodyCap {

namespace Probe = JoltSoftBodyCapProbe;

constexpr const char *K_CONFIG = "rod/config";
constexpr const char *K_STATE = "rod/state";

// Non-SceneTree/Editor tests start without a physics singleton. Restore it even
// when REQUIRE aborts. Cache one server per name: per-case init/finish corrupts
// Jolt's global job/Factory/RegisterTypes state. Isolate cases with fresh spaces
// and bodies instead.
class ServerScope {
	PhysicsServer3D *saved = nullptr;
	PhysicsServer3D *server = nullptr;

public:
	explicit ServerScope(const String &p_name) {
		saved = PhysicsServer3D::get_singleton();

		static HashMap<String, PhysicsServer3D *> servers;
		if (HashMap<String, PhysicsServer3D *>::Iterator found = servers.find(p_name)) {
			server = found->value;
		} else {
			server = PhysicsServer3DManager::get_singleton()->new_server(p_name);
			if (server != nullptr) {
				server->init();
				server->set_active(true);
			}
			servers.insert(p_name, server);
		}

		// Reuse does not run the constructor that claimed the singleton.
		if (server != nullptr) {
			PhysicsServer3D::set_singleton_for_tests(server);
		}
	}

	~ServerScope() {
		PhysicsServer3D::set_singleton_for_tests(saved);
	}

	bool is_valid() const { return server != nullptr; }
	PhysicsServer3D *ptr() const { return server; }
	PhysicsServer3D *operator->() const { return server; }
};

// Capture offending values in diagnostics while suppressing expected error output.
// Error handlers still run with printing disabled.
class ErrorCapture {
	static void _handle(void *p_userdata, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, bool p_editor_notify, ErrorHandlerType p_type) {
		ErrorCapture *capture = static_cast<ErrorCapture *>(p_userdata);
		capture->messages.push_back(String::utf8(p_error) + " " + String::utf8(p_message));
		if (p_type == ERR_HANDLER_WARNING) {
			++capture->warnings;
		}
	}

	ErrorHandlerList list;
	Vector<String> messages;
	int warnings = 0;

public:
	ErrorCapture() {
		list.errfunc = &_handle;
		list.userdata = this;
		add_error_handler(&list);
		ERR_PRINT_OFF;
	}

	~ErrorCapture() {
		ERR_PRINT_ON;
		remove_error_handler(&list);
	}

	void clear() {
		messages.clear();
		warnings = 0;
	}
	int count() const { return messages.size(); }
	int warning_count() const { return warnings; }

	bool has(const String &p_fragment) const {
		for (const String &message : messages) {
			if (message.contains(p_fragment)) {
				return true;
			}
		}
		return false;
	}

	String joined() const {
		String out;
		for (const String &message : messages) {
			out += message + "\n";
		}
		return out;
	}
};

// A space that can be stepped. Order matches `Main::iteration`.
class SpaceScope {
	PhysicsServer3D *server = nullptr;
	RID space;

public:
	explicit SpaceScope(PhysicsServer3D *p_server) :
			server(p_server) {
		space = server->space_create();
		server->space_set_active(space, true);
	}

	~SpaceScope() { server->free_rid(space); }

	RID rid() const { return space; }

	void step(int p_frames = 1) {
		for (int i = 0; i < p_frames; i++) {
			server->sync();
			server->flush_queries();
			server->end_sync();
			server->step(1.0 / 60.0);
		}
	}
};

// Unit-spaced +X chain: unit segment lengths and identical Bishop frames.
PackedVector3Array make_joints(int p_count, real_t p_spacing = 1.0) {
	PackedVector3Array joints;
	for (int i = 0; i < p_count; i++) {
		joints.push_back(Vector3(p_spacing * i, 0, 0));
	}
	return joints;
}

PackedFloat32Array make_floats(int p_count, float p_value = 0.0f) {
	PackedFloat32Array values;
	for (int i = 0; i < p_count; i++) {
		values.push_back(p_value);
	}
	return values;
}

Dictionary rod_config(int p_joints, int p_fixed = 1) {
	Dictionary config;
	config["joints"] = make_joints(p_joints);
	config["compliance"] = make_floats(p_joints - 1);
	config["bend"] = make_floats(MAX(p_joints - 2, 0));
	config["fixed"] = p_fixed;
	return config;
}

// Build a complete candidate locally. No partial writes reach the server.
Dictionary rod_candidate(PhysicsServer3D *p_server, RID p_body, const String &p_field, const Variant &p_value) {
	const Variant stored = p_server->soft_body_get_extra_property(p_body, K_CONFIG);
	Dictionary config = stored.get_type() == Variant::DICTIONARY ? Dictionary(stored) : Dictionary();
	config[p_field] = p_value;
	return config;
}

bool set_valid_rod(PhysicsServer3D *p_server, RID p_body, int p_joints, int p_fixed = 1) {
	return p_server->soft_body_set_extra_property(p_body, K_CONFIG, rod_config(p_joints, p_fixed));
}

// Two triangles sharing an edge: the smallest thing the cloth path accepts.
RID make_quad_mesh() {
	RenderingServer *rendering = RenderingServer::get_singleton();

	PackedVector3Array vertices;
	vertices.push_back(Vector3(0, 0, 0));
	vertices.push_back(Vector3(1, 0, 0));
	vertices.push_back(Vector3(1, 0, 1));
	vertices.push_back(Vector3(0, 0, 1));

	PackedInt32Array indices;
	const int triangles[6] = { 0, 1, 2, 0, 2, 3 };
	for (int index : triangles) {
		indices.push_back(index);
	}

	Array arrays;
	arrays.resize(RSE::ARRAY_MAX);
	arrays[RSE::ARRAY_VERTEX] = vertices;
	arrays[RSE::ARRAY_INDEX] = indices;

	RID mesh = rendering->mesh_create();
	rendering->mesh_add_surface_from_arrays(mesh, RSE::PRIMITIVE_TRIANGLES, arrays);
	return mesh;
}

Dictionary find_property(const TypedArray<Dictionary> &p_list, const String &p_name) {
	for (int i = 0; i < p_list.size(); i++) {
		const Dictionary entry = p_list[i];
		if (String(entry["name"]) == p_name) {
			return entry;
		}
	}
	return Dictionary();
}

// The live rotation of rod `p_index`, read out of `rod/state`.
Quaternion read_rod_rotation(PhysicsServer3D *p_server, RID p_body, int p_index) {
	const PackedFloat32Array state = p_server->soft_body_get_extra_property(p_body, K_STATE);
	if (state.size() < (p_index + 1) * 4) {
		return Quaternion(0, 0, 0, 0);
	}
	return Quaternion(state[p_index * 4 + 0], state[p_index * 4 + 1], state[p_index * 4 + 2], state[p_index * 4 + 3]);
}

// Exact config rejection oracle shared by all new-capability error tables.
class RemainingSnapshot {
	PhysicsServer3D *server;
	RID body;
	Array values;
	Variant rod_state;
	Vector<Vector3> positions, velocities;
	Vector<float> shared_masses, runtime_masses;
	uint64_t identity, settings, generation;
	float mass, stiffness;
	int precision;

public:
	RemainingSnapshot(PhysicsServer3D *p_server, RID p_body) : server(p_server), body(p_body) {
		for (const char *key : { "scalar/config", "bend/config", "lra/config", "vertex/config", "volume/config", "skin/config", "rod/config", "skin/pose" }) {
			values.push_back(server->soft_body_get_extra_property(body, key));
		}
		if (Probe::in_space(body)) {
			rod_state = server->soft_body_get_extra_property(body, "rod/state");
		}
		identity = Probe::body_identity(body);
		settings = Probe::settings_identity(body);
		generation = Probe::build_counts(body).generation;
		for (int i = 0; i < Probe::vertex_count(body); ++i) {
			positions.push_back(Probe::world_position(body, i));
			velocities.push_back(Probe::velocity(body, i));
			shared_masses.push_back(Probe::shared_inv_mass(body, i));
			runtime_masses.push_back(Probe::vertex_inv_mass(body, i));
		}
		mass = server->soft_body_get_total_mass(body);
		stiffness = server->soft_body_get_linear_stiffness(body);
		precision = server->soft_body_get_simulation_precision(body);
	}
	void unchanged() const {
		int i = 0;
		for (const char *key : { "scalar/config", "bend/config", "lra/config", "vertex/config", "volume/config", "skin/config", "rod/config", "skin/pose" }) {
			CAPTURE(key);
			CHECK(server->soft_body_get_extra_property(body, key) == values[i++]);
		}
		if (Probe::in_space(body)) {
			CHECK(server->soft_body_get_extra_property(body, "rod/state") == rod_state);
		}
		CHECK(Probe::body_identity(body) == identity);
		CHECK(Probe::settings_identity(body) == settings);
		CHECK(Probe::build_counts(body).generation == generation);
		for (int vertex = 0; vertex < positions.size(); ++vertex) {
			CHECK(Probe::world_position(body, vertex) == positions[vertex]);
			CHECK(Probe::velocity(body, vertex) == velocities[vertex]);
			CHECK(Probe::shared_inv_mass(body, vertex) == shared_masses[vertex]);
			CHECK(Probe::vertex_inv_mass(body, vertex) == runtime_masses[vertex]);
		}
		CHECK(server->soft_body_get_total_mass(body) == mass);
		CHECK(server->soft_body_get_linear_stiffness(body) == stiffness);
		CHECK(server->soft_body_get_simulation_precision(body) == precision);
	}
};

struct RemainingMeshFixture {
	PackedVector3Array vertices;
	PackedInt32Array indices;
	PackedInt32Array pins;

	RID create_mesh() const {
		Array arrays;
		arrays.resize(RSE::ARRAY_MAX);
		arrays[RSE::ARRAY_VERTEX] = vertices;
		arrays[RSE::ARRAY_INDEX] = indices;
		RID mesh = RenderingServer::get_singleton()->mesh_create();
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(mesh, RSE::PRIMITIVE_TRIANGLES, arrays);
		return mesh;
	}
};

RemainingMeshFixture fixture_grid(int p_side = 21) {
	RemainingMeshFixture result;
	for (int z = 0; z < p_side; ++z) {
		for (int x = 0; x < p_side; ++x) {
			result.vertices.push_back(Vector3(float(x) / (p_side - 1) - 0.5f, 0, float(z) / (p_side - 1)));
			if (z < 2) {
				result.pins.push_back(z * p_side + x);
			}
		}
	}
	for (int z = 0; z + 1 < p_side; ++z) {
		for (int x = 0; x + 1 < p_side; ++x) {
			const int a = z * p_side + x;
			for (int index : { a, a + 1, a + p_side + 1, a, a + p_side + 1, a + p_side }) {
				result.indices.push_back(index);
			}
		}
	}
	return result;
}

RemainingMeshFixture fixture_hinge(bool p_folded = false) {
	RemainingMeshFixture result;
	for (const Vector3 &vertex : { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1), (p_folded ? Vector3(0.5f, 0.6f, 0.5f) : Vector3(-0.7f, 0, 0.4f)) }) {
		result.vertices.push_back(vertex);
	}
	for (int index : { 0, 1, 2, 0, 2, 3 }) {
		result.indices.push_back(index);
	}
	return result;
}

RemainingMeshFixture fixture_seam() {
	RemainingMeshFixture result = fixture_hinge();
	// Source 4 aliases source 0, source 5 aliases source 2. Source 6 is unused.
	result.vertices.push_back(result.vertices[0]);
	result.vertices.push_back(result.vertices[2]);
	result.vertices.push_back(Vector3(7, 8, 9));
	result.indices.set(3, 4);
	result.indices.set(4, 5);
	return result;
}

RemainingMeshFixture fixture_graph(bool p_disconnected = false, bool p_two_pins = false) {
	RemainingMeshFixture result;
	const Vector3 centers[] = { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(2, 0, 0), Vector3(2, 0, 1), Vector3(1, 0, 1), Vector3(0, 0, 1) };
	for (const Vector3 &center : centers) {
		result.vertices.push_back(center);
		result.vertices.push_back(center + Vector3(0, 0.1f, 0));
	}
	for (int i = 0; i < 5; ++i) {
		for (int index : { 2 * i, 2 * i + 1, 2 * i + 3, 2 * i, 2 * i + 3, 2 * i + 2 }) {
			result.indices.push_back(index);
		}
	}
	result.pins.push_back(0);
	if (p_two_pins) {
		result.pins.push_back(11);
	}
	if (p_disconnected) {
		const int offset = result.vertices.size();
		for (const Vector3 &vertex : { Vector3(4, 0, 0), Vector3(5, 0, 0), Vector3(4, 0, 1) }) {
			result.vertices.push_back(vertex);
		}
		for (int index : { offset, offset + 1, offset + 2 }) {
			result.indices.push_back(index);
		}
	}
	return result;
}

Dictionary fixture_tetra(bool p_pair = false) {
	Dictionary result;
	PackedVector3Array vertices;
	for (const Vector3 &vertex : { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) }) {
		vertices.push_back(vertex);
	}
	PackedInt32Array tetrahedra;
	for (int index : { 0, 1, 2, 3 }) {
		tetrahedra.push_back(index);
	}
	if (p_pair) {
		vertices.push_back(Vector3(0, 0, -1));
		for (int index : { 0, 2, 1, 4 }) {
			tetrahedra.push_back(index);
		}
	}
	result["vertices"] = vertices;
	result["tetrahedra"] = tetrahedra;
	return result;
}

Dictionary fixture_skin(int p_joints = 1, int p_selected = 4) {
	Dictionary result;
	PackedInt32Array vertices;
	PackedInt32Array indices;
	PackedFloat32Array weights;
	for (int vertex = 0; vertex < p_selected; ++vertex) {
		vertices.push_back(vertex);
		for (int slot = 0; slot < 4; ++slot) {
			indices.push_back(slot < p_joints ? slot : 0);
			weights.push_back(slot < p_joints ? 1.0f / p_joints : 0.0f);
		}
	}
	Array inv_bind;
	Array pose;
	for (int i = 0; i < p_joints; ++i) {
		inv_bind.push_back(Transform3D());
		pose.push_back(Transform3D());
	}
	result["vertices"] = vertices;
	result["joint_indices"] = indices;
	result["joint_weights"] = weights;
	result["inv_bind"] = inv_bind;
	result["initial_pose"] = pose;
	return result;
}

const char *const REMAINING_CONFIG_KEYS[] = { "scalar/config", "bend/config", "lra/config", "vertex/config", "volume/config", "skin/config", "rod/config" };
const char *const REMAINING_LIVE_KEYS[] = { "scalar/current", "bend/count", "lra/count", "volume/count", "skin/count", "volume/current", "volume/positions", "volume/faces", "skin/frame", "rod/state" };
Dictionary remaining_config(int p_cap) {
	Dictionary config;
	switch (p_cap) {
		case 0:
			config["friction"] = 0.4;
			config["restitution"] = 0.2;
			break;
		case 1:
			config["compliance"] = 1e-4;
			break;
		case 2:
			config["type"] = 1;
			break;
		case 3:
			config["edge_scale"] = 1.0;
			config["shear_scale"] = 1.0;
			break;
		case 4:
			return fixture_tetra();
		case 5:
			return fixture_skin();
		case 6:
			return rod_config(4);
	}
	return config;
}

class RemainingAreaScope {
	PhysicsServer3D *server;
	RID shape;
	RID area;
	bool wind;
	class Monitor : public Object {
	public:
		int entered = 0;
		void event(int p_status, RID, ObjectID, int, int) {
			if (p_status == PS3DE::AREA_BODY_ADDED) {
				++entered;
			}
		}
	};
	Monitor *monitor = memnew(Monitor);

public:
	int entered_count() const { return monitor->entered; }
	RemainingAreaScope(PhysicsServer3D *p_server, RID p_space, bool p_wind) :
			server(p_server), shape(server->shape_create(PS3DE::SHAPE_BOX)), area(server->area_create()), wind(p_wind) {
		server->shape_set_data(shape, Vector3(100, 100, 100));
		server->area_add_shape(area, shape);
		server->area_set_param(area, PS3DE::AREA_PARAM_GRAVITY_OVERRIDE_MODE, PS3DE::AREA_SPACE_OVERRIDE_REPLACE);
		server->area_set_param(area, PS3DE::AREA_PARAM_GRAVITY, 0.0);
		// Install the body-monitor callback normally supplied by Area3D.
		server->area_set_monitor_callback(area, callable_mp(monitor, &Monitor::event));
		server->area_set_space(area, p_space);
	}
	void enable_forces() {
		server->area_set_param(area, PS3DE::AREA_PARAM_GRAVITY, wind ? 0.0 : 3.0);
		server->area_set_param(area, PS3DE::AREA_PARAM_GRAVITY_VECTOR, Vector3(0, 1, 0));
		if (wind) {
			server->area_set_param(area, PS3DE::AREA_PARAM_WIND_FORCE_MAGNITUDE, 2.0);
			server->area_set_param(area, PS3DE::AREA_PARAM_WIND_SOURCE, Vector3(0, -10, 0));
			server->area_set_param(area, PS3DE::AREA_PARAM_WIND_DIRECTION, Vector3(0, 1, 0));
			server->area_set_param(area, PS3DE::AREA_PARAM_WIND_ATTENUATION_FACTOR, 0.0);
		}
	}

	~RemainingAreaScope() {
		server->free_rid(area);
		server->free_rid(shape);
		memdelete(monitor);
	}
};

// Static contact fixture. Owns both collision body and shape.
class RemainingFloorScope {
	PhysicsServer3D *server;
	RID shape;
	RID body;

public:
	RemainingFloorScope(PhysicsServer3D *p_server, RID p_space, float p_slope = 0.0f) :
			server(p_server), shape(server->shape_create(PS3DE::SHAPE_BOX)), body(server->body_create()) {
		server->shape_set_data(shape, Vector3(20, 0.1f, 20));
		server->body_set_mode(body, PS3DE::BODY_MODE_STATIC);
		server->body_add_shape(body, shape);
		const Basis rotation(Vector3(0, 0, 1), p_slope);
		server->body_set_state(body, PS3DE::BODY_STATE_TRANSFORM, Transform3D(rotation, rotation.xform(Vector3(0, -0.1f, 0))));
		server->body_set_param(body, PS3DE::BODY_PARAM_FRICTION, 1.0);
		server->body_set_param(body, PS3DE::BODY_PARAM_BOUNCE, 0.0);
		server->body_set_space(body, p_space);
	}
	~RemainingFloorScope() {
		server->free_rid(body);
		server->free_rid(shape);
	}
};

// Self-supplied volume fixture owns no RenderingServer resource.
class VolumeBodyScope {
	PhysicsServer3D *server;
	RID body;

public:
	VolumeBodyScope(PhysicsServer3D *p_server, RID p_space, const Dictionary &p_config, bool p_control = false) : server(p_server), body(server->soft_body_create()) {
		server->soft_body_set_simulation_precision(body, 20);
		server->soft_body_set_total_mass(body, 1.0);
		server->soft_body_set_pressure_coefficient(body, 0.0);
		server->soft_body_set_state(body, PS3DE::BODY_STATE_CAN_SLEEP, false);
		if (p_control) {
			REQUIRE(Probe::volume_control(body));
		}
		REQUIRE(server->soft_body_set_extra_property(body, "volume/config", p_config));
		server->soft_body_set_space(body, p_space);
		REQUIRE(Probe::in_space(body));
	}
	~VolumeBodyScope() { server->free_rid(body); }
	RID rid() const { return body; }
};

// Owns renderer input and physics body. Body dies before mesh and space.
class MeshBodyScope {
	PhysicsServer3D *server;
	RID body;
	RID mesh;

public:
	explicit MeshBodyScope(PhysicsServer3D *p_server, RID p_space, const RemainingMeshFixture *p_fixture = nullptr) :
			server(p_server), body(server->soft_body_create()), mesh(p_fixture == nullptr ? make_quad_mesh() : p_fixture->create_mesh()) {
		server->soft_body_set_simulation_precision(body, 20);
		server->soft_body_set_total_mass(body, 1.0f);
		server->soft_body_set_pressure_coefficient(body, 0.0f);
		server->soft_body_set_state(body, PS3DE::BODY_STATE_CAN_SLEEP, false);
		if (p_fixture != nullptr) {
			for (int pin : p_fixture->pins) {
				server->soft_body_pin_point(body, pin, true);
			}
		}
		server->soft_body_set_mesh(body, mesh);
		server->soft_body_set_space(body, p_space);
	}
	~MeshBodyScope() {
		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}
	RID rid() const { return body; }
	RID mesh_rid() const { return mesh; }
};

} // namespace TestJoltSoftBodyCap
