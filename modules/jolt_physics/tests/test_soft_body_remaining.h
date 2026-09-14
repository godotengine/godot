/**************************************************************************/
/*  test_soft_body_remaining.h                                            */
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

#include "../capabilities/soft_body_cap_validation.h"
#include "soft_body_cap_test_support.h"

#include "core/io/file_access.h"
#include "core/io/json.h"
#include "servers/physics_3d/physics_server_3d_rendering_server_handler.h"
#include "servers/physics_3d/physics_server_3d_wrap_mt.h"
#include "tests/test_utils.h"

#include <cfloat>
#include <cmath>
#include <limits>

namespace TestJoltSoftBodyCap {
TEST_SUITE("[JoltSoftBodyRemaining]") {
	TEST_CASE("remaining P01 unknown prefix") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, "nope/x", 1));
		CHECK(errors.count() == 0);
		server->free_rid(body);
	}

	TEST_CASE("remaining B01 default cloth") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int variant = 0; variant < 3; ++variant) {
			CAPTURE(variant);
			MeshBodyScope body(server.ptr(), space.rid());
			const float mass[] = { 1.0f, 2.0f, 0.5f };
			const float pressure[] = { 0.0f, 0.2f, 1.0f };
			const float stiffness[] = { 0.1f, 0.5f, 1.0f };
			server->soft_body_set_total_mass(body.rid(), mass[variant]);
			server->soft_body_set_pressure_coefficient(body.rid(), pressure[variant]);
			server->soft_body_set_linear_stiffness(body.rid(), stiffness[variant]);
			server->soft_body_set_space(body.rid(), RID());
			server->soft_body_set_space(body.rid(), space.rid());
			REQUIRE(Probe::in_space(body.rid()));
			CHECK(Probe::vertex_count(body.rid()) == 4);
			CHECK(Probe::edge_count(body.rid()) > 0);
			CHECK(Probe::dihedral_bend_count(body.rid()) == 0);
			CHECK(Probe::lra_count(body.rid()) == 0);
			CHECK(Probe::volume_count(body.rid()) == 0);
			CHECK(Probe::skinned_count(body.rid()) == 0);
			CHECK(Probe::inv_bind_count(body.rid()) == 0);
			CHECK(Probe::all_finite(body.rid()));
			CHECK(Probe::settings_identity(body.rid()) != 0);
			CHECK(Probe::body_identity(body.rid()) != UINT64_MAX);
			CHECK(Probe::edge(body.rid(), 0).valid);
			CHECK_FALSE(Probe::edge(body.rid(), -1).valid);
			CHECK_FALSE(Probe::edge(body.rid(), Probe::edge_count(body.rid())).valid);
			CHECK_FALSE(Probe::dihedral(body.rid(), 0).valid);
			CHECK_FALSE(Probe::lra(body.rid(), 0).valid);
			CHECK_FALSE(Probe::tetra(body.rid(), 0).valid);
			CHECK_FALSE(Probe::skin(body.rid(), 0).valid);
			CHECK(Probe::shared_inv_mass(body.rid(), -1) == -1.0f);
			const auto counts = Probe::build_counts(body.rid());
			CHECK(counts.generation == 2);
			CHECK(counts.create_constraints == 2);
			CHECK(counts.optimize == 2);
		}
		CHECK_FALSE(Probe::edge(RID(), 0).valid);
		CHECK_FALSE(Probe::all_finite(RID()));
		CHECK(Probe::body_identity(RID()) == UINT64_MAX);

		// Qualify Q/H against real Jolt generator before Distance behavior tests.
		int edges = -1;
		int dihedrals = -1;
		CHECK(Probe::fixture_bend(true, 0, edges, dihedrals).valid);
		CHECK(edges == 6);
		CHECK(dihedrals == 0);
		Probe::fixture_bend(true, 2, edges, dihedrals);
		CHECK(dihedrals == 1);
		CHECK_FALSE(Probe::fixture_bend(false, 0, edges, dihedrals).valid);
		CHECK(edges == 5);
		const auto distance = Probe::fixture_bend(false, 1, edges, dihedrals);
		CHECK(distance.valid);
		CHECK(edges == 6);
		CHECK(dihedrals == 0);
		CHECK(distance.compliance == doctest::Approx(1.0e-4f));
		CHECK(distance.rest == doctest::Approx(Vector3(1.7f, 0, -0.4f).length()));
	}

	TEST_CASE("remaining S01 scalar fields") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		// Stock first creation and space replay use different radius sources.
		// Qualify the rebuilt control before applying overrides.
		const auto fresh = Probe::scalars(body.rid());
		REQUIRE(fresh.valid);
		CHECK(fresh.vertex_radius == 0.0f);
		server->soft_body_set_space(body.rid(), RID());
		server->soft_body_set_space(body.rid(), space.rid());
		const auto rebuilt = Probe::scalars(body.rid());
		REQUIRE(rebuilt.valid);
		const float global_radius = GLOBAL_GET("physics/jolt_physics_3d/simulation/soft_body_point_radius");
		CAPTURE(fresh.vertex_radius);
		CAPTURE(rebuilt.vertex_radius);
		CAPTURE(global_radius);
		CHECK(rebuilt.vertex_radius == global_radius);
		const Dictionary defaults = server->soft_body_get_extra_property(body.rid(), "scalar/current");
		REQUIRE(defaults.size() == 5);
		const char *fields[] = { "friction", "restitution", "gravity_factor", "faces_double_sided", "vertex_radius" };
		const Variant values[] = { 0.3, 0.7, 2.0, true, 0.1 };
		for (int field = 0; field < 5; ++field) {
			CAPTURE(fields[field]);
			for (int mode = 0; mode < 2; ++mode) {
				CAPTURE(mode);
				Dictionary config;
				config[fields[field]] = mode == 0 ? defaults[fields[field]] : values[field];
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
				const Dictionary actual = server->soft_body_get_extra_property(body.rid(), "scalar/current");
				REQUIRE(actual.size() == 5);
				for (const char *other : fields) {
					const Variant expected = String(other) == fields[field] ? config[other] : defaults[other];
					if (expected.get_type() == Variant::BOOL) {
						CHECK(bool(actual[other]) == bool(expected));
					} else {
						CHECK(double(actual[other]) == doctest::Approx(double(expected)).epsilon(1e-5));
					}
				}
				const auto probe = Probe::scalars(body.rid());
				CHECK(probe.valid);
				CHECK(probe.friction == float(actual["friction"]));
				CHECK(probe.restitution == float(actual["restitution"]));
				CHECK(probe.gravity_factor == float(actual["gravity_factor"]));
				CHECK(probe.vertex_radius == float(actual["vertex_radius"]));
				CHECK(probe.faces_double_sided == bool(actual["faces_double_sided"]));
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", Variant()));
				const Dictionary restored = server->soft_body_get_extra_property(body.rid(), "scalar/current");
				CHECK(restored == defaults);
			}
		}
	}

	TEST_CASE("remaining S02 gravity behavior") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY_VECTOR, Vector3(0, -1, 0));
		float drops[3];
		for (int factor = 0; factor < 3; ++factor) {
			CAPTURE(factor);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config;
			config["gravity_factor"] = double(factor);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
			const Vector3 before = Probe::world_position(body.rid(), 0);
			space.step(60);
			REQUIRE(Probe::all_finite(body.rid()));
			drops[factor] = before.y - Probe::world_position(body.rid(), 0).y;
		}
		CHECK(Math::abs(drops[0]) < 1e-5f);
		CHECK(drops[1] > 0.001f);
		CHECK(drops[2] > drops[1] + 0.001f);
		CHECK(drops[2] == doctest::Approx(2 * drops[1]).epsilon(1e-4));
	}

	TEST_CASE("remaining S03 gravity neighbors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (int mode = 0; mode < 3; ++mode) {
			CAPTURE(mode);
			Vector3 displacement[3];
			for (int factor = 0; factor < 3; ++factor) {
				CAPTURE(factor);
				SpaceScope space(server.ptr());
				server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
				RemainingAreaScope area(server.ptr(), space.rid(), mode != 0);
				MeshBodyScope body(server.ptr(), space.rid());
				Dictionary config;
				config["gravity_factor"] = double(factor);
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
				// Establish sensor overlap before observing area override / wind.
				space.step(2);
				REQUIRE(area.entered_count() == 1);
				area.enable_forces();
				server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
				// Reset velocity after overlap establishment; controls start at rest.
				server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D());
				if (mode == 2) {
					server->soft_body_apply_central_impulse(body.rid(), Vector3(1, 0, 0));
				}
				// Average all particles: internal edge corrections can move one
				// particle but cannot change total momentum from environmental force.
				Vector3 start;
				for (int vertex = 0; vertex < Probe::vertex_count(body.rid()); ++vertex) {
					start += Probe::world_position(body.rid(), vertex);
				}
				space.step(30);
				Vector3 finish;
				for (int vertex = 0; vertex < Probe::vertex_count(body.rid()); ++vertex) {
					finish += Probe::world_position(body.rid(), vertex);
				}
				displacement[factor] = (finish - start) / Probe::vertex_count(body.rid());
				REQUIRE(Probe::all_finite(body.rid()));
			}
			CAPTURE(displacement[0]);
			CAPTURE(displacement[1]);
			CAPTURE(displacement[2]);
			if (mode == 0) {
				// Upward area gravity opposes downward default gravity. Translation
				// normal to the cloth avoids in-plane edge roundoff in this force test.
				CHECK(Math::abs(displacement[0].y) < 1e-5f);
				CHECK(displacement[1].y > 1e-3f);
				CHECK(displacement[2].y == doctest::Approx(2 * displacement[1].y).epsilon(1e-4));
			} else {
				CHECK(displacement[0].y > 1e-3f);
				CHECK(displacement[1].y == doctest::Approx(displacement[0].y).epsilon(1e-4));
				CHECK(displacement[2].y == doctest::Approx(displacement[0].y).epsilon(1e-4));
				if (mode == 2) {
					CHECK(displacement[0].x > 1e-3f);
					CHECK(displacement[1].x == doctest::Approx(displacement[0].x).epsilon(1e-4));
					CHECK(displacement[2].x == doctest::Approx(displacement[0].x).epsilon(1e-4));
				}
			}
		}
	}

	TEST_CASE("remaining S04 friction contact") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		float slides[2];
		for (int friction = 0; friction <= 1; ++friction) {
			CAPTURE(friction);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			RemainingFloorScope floor(server.ptr(), space.rid(), 0.25f);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config;
			config["friction"] = double(friction);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
			server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D(Basis(Vector3(0, 0, 1), 0.25f), Vector3(0, 0.5f, 0)));
			const float initial_x = Probe::world_position(body.rid(), 0).x;
			Probe::ContactScope contact_probe(body.rid());
			int contacts = 0;
			for (int frame = 0; frame < 120; ++frame) {
				space.step();
				contacts = contact_probe.count();
				REQUIRE(Probe::all_finite(body.rid()));
			}
			CAPTURE(contacts);
			REQUIRE(contacts > 0);
			slides[friction] = initial_x - Probe::world_position(body.rid(), 0).x;
		}
		CAPTURE(slides[0]);
		CAPTURE(slides[1]);
		CHECK(slides[0] > slides[1] + 1e-3f);
	}

	TEST_CASE("remaining S05 restitution contact") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		float peaks[2] = {};
		for (int restitution = 0; restitution <= 1; ++restitution) {
			CAPTURE(restitution);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			RemainingFloorScope floor(server.ptr(), space.rid());
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config;
			config["restitution"] = double(restitution);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
			server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D(Basis(), Vector3(0, 0.5f, 0)));
			server->soft_body_apply_central_impulse(body.rid(), Vector3(0, -2, 0));
			Probe::ContactScope contact_probe(body.rid());
			int contacts = 0;
			for (int frame = 0; frame < 120; ++frame) {
				space.step();
				contacts = contact_probe.count();
				REQUIRE(Probe::all_finite(body.rid()));
				if (contacts > 0) {
					peaks[restitution] = MAX(peaks[restitution], Probe::world_position(body.rid(), 0).y);
				}
			}
			CAPTURE(contacts);
			REQUIRE(contacts > 0);
		}
		CAPTURE(peaks[0]);
		CAPTURE(peaks[1]);
		CHECK(peaks[1] > peaks[0] + 1e-3f);
	}

	TEST_CASE("remaining S06 face sidedness") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (bool sided : { false, true, false }) {
			CAPTURE(sided);
			Dictionary config;
			config["faces_double_sided"] = sided;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
			CHECK(Probe::scalars(body.rid()).faces_double_sided == sided);
			// Settings winding (2,1,0) points +Y; exercise SoftBodyShape's
			// collector overload, not Godot's filter which excludes soft bodies.
			CHECK(Probe::face_ray_hit(body.rid(), Vector3(0.25f, 1, 0.5f), Vector3(0, -2, 0)));
			CHECK(Probe::face_ray_hit(body.rid(), Vector3(0.25f, -1, 0.5f), Vector3(0, 2, 0)) == sided);
		}
		CHECK_FALSE(Probe::face_ray_hit(RID(), Vector3(), Vector3(0, 1, 0)));
	}

	TEST_CASE("remaining S07 radius contact") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		const float global_radius = GLOBAL_GET("physics/jolt_physics_3d/simulation/soft_body_point_radius");
		for (int mode = 0; mode < 3; ++mode) {
			CAPTURE(mode);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
			RemainingFloorScope floor(server.ptr(), space.rid());
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config;
			config["gravity_factor"] = 0.0;
			if (mode != 0) {
				config["vertex_radius"] = mode == 1 ? 0.0 : 0.1;
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
			server->soft_body_set_space(body.rid(), RID());
			server->soft_body_set_space(body.rid(), space.rid());
			CHECK(Probe::scalars(body.rid()).vertex_radius == doctest::Approx(mode == 0 ? global_radius : mode == 1 ? 0.0f
																													: 0.1f));
			server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D(Basis(), Vector3(0, 0.05f, 0)));
			Probe::ContactScope contact_probe(body.rid());
			int contacts = 0;
			for (int frame = 0; frame < 3; ++frame) {
				space.step();
				contacts = contact_probe.count();
			}
			CAPTURE(contacts);
			REQUIRE(Probe::all_finite(body.rid()));
			if (mode == 2) {
				CHECK(contacts > 0);
				CHECK(Probe::world_position(body.rid(), 0).y > 0.05f + 1e-3f);
			} else {
				CHECK(contacts == 0);
				CHECK(Probe::world_position(body.rid(), 0).y == doctest::Approx(0.05f).epsilon(1e-5));
			}
		}
		CHECK(Probe::ContactScope(RID()).count() == -1);
	}

	TEST_CASE("remaining S08 scalar lifetime") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr());
		SpaceScope second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		server->soft_body_set_space(body.rid(), RID());
		server->soft_body_set_space(body.rid(), first.rid());
		const Dictionary defaults = server->soft_body_get_extra_property(body.rid(), "scalar/current");
		Dictionary config;
		config["friction"] = 0.23;
		config["restitution"] = 0.71;
		config["gravity_factor"] = 0.0;
		config["vertex_radius"] = 0.14;
		config["faces_double_sided"] = true;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
		const Dictionary expected = server->soft_body_get_extra_property(body.rid(), "scalar/current");
		RID next_mesh = make_quad_mesh();
		for (int change = 0; change < 3; ++change) {
			CAPTURE(change);
			if (change == 0) {
				server->soft_body_set_mesh(body.rid(), next_mesh);
			} else if (change == 1) {
				server->soft_body_set_space(body.rid(), second.rid());
			} else {
				server->soft_body_set_total_mass(body.rid(), 2.0);
			}
			REQUIRE(Probe::in_space(body.rid()));
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/current")) == expected);
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/config")) == config);
		}
		server->soft_body_set_mesh(body.rid(), body.mesh_rid());
		RenderingServer::get_singleton()->free_rid(next_mesh);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", Variant()));
		CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/current")) == defaults);
	}

	TEST_CASE("remaining B02 dihedral topology") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (bool explicit_type : { false, true }) {
			CAPTURE(explicit_type);
			Dictionary config;
			config["compliance"] = 1e-4;
			if (explicit_type) {
				config["type"] = 1;
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
			CHECK(int(server->soft_body_get_extra_property(body.rid(), "bend/count")) == 1);
			REQUIRE(Probe::dihedral_bend_count(body.rid()) == 1);
			const auto bend = Probe::dihedral(body.rid(), 0);
			CHECK(bend.vertices[0] == 0);
			CHECK(bend.vertices[1] == 2);
			CHECK(((bend.vertices[2] == 1 && bend.vertices[3] == 3) || (bend.vertices[2] == 3 && bend.vertices[3] == 1)));
			CHECK(bend.compliance == doctest::Approx(1e-4).epsilon(1e-5));
		}
	}

	TEST_CASE("remaining B03 distance topology") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_hinge();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		REQUIRE(Probe::edge_count(body.rid()) == 5);
		Dictionary config;
		config["compliance"] = 1e-4;
		config["type"] = 0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
		CHECK(int(server->soft_body_get_extra_property(body.rid(), "bend/count")) == 0);
		CHECK(Probe::dihedral_bend_count(body.rid()) == 0);
		CHECK(Probe::edge_count(body.rid()) == 6);
		int matched = 0;
		for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
			const auto edge = Probe::edge(body.rid(), i);
			if (MIN(edge.vertices[0], edge.vertices[1]) == 1 && MAX(edge.vertices[0], edge.vertices[1]) == 3) {
				++matched;
				CHECK(edge.rest == doctest::Approx(fixture.vertices[1].distance_to(fixture.vertices[3])).epsilon(1e-5));
				CHECK(edge.compliance == doctest::Approx(1e-4).epsilon(1e-5));
			}
		}
		CHECK(matched == 1);
	}

	TEST_CASE("remaining B04 bend gates") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (int mask = 0; mask < 4; ++mask) {
			for (double finite : { 0.0, 1e-4 }) {
				CAPTURE(mask);
				CAPTURE(finite);
				PackedFloat32Array values = make_floats(4, finite);
				values.set(0, (mask & 1) ? FLT_MAX : finite);
				values.set(2, (mask & 2) ? FLT_MAX : finite);
				Dictionary config;
				config["compliance"] = values;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
				CHECK(Probe::dihedral_bend_count(body.rid()) == (mask == 0 ? 1 : 0));
				if (mask == 0) {
					CHECK(Probe::dihedral(body.rid(), 0).compliance == doctest::Approx(finite).epsilon(1e-5));
				}
			}
		}
	}

	TEST_CASE("remaining B05 rest angle") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool folded : { false, true }) {
			CAPTURE(folded);
			const auto fixture = fixture_hinge(folded);
			MeshBodyScope body(server.ptr(), space.rid(), &fixture);
			Dictionary config;
			config["compliance"] = 1e-4;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
			REQUIRE(Probe::dihedral_bend_count(body.rid()) == 1);
			const auto bend = Probe::dihedral(body.rid(), 0);
			const Vector3 a = Probe::vertex_position(body.rid(), bend.vertices[0]);
			const Vector3 e = Probe::vertex_position(body.rid(), bend.vertices[1]) - a;
			const Vector3 n1 = e.cross(Probe::vertex_position(body.rid(), bend.vertices[2]) - a).normalized();
			const Vector3 n2 = (Probe::vertex_position(body.rid(), bend.vertices[3]) - a).cross(e).normalized();
			const double angle = Math::acos(CLAMP(double(n1.dot(n2)), -1.0, 1.0)) * SIGN(n2.cross(n1).dot(e));
			CAPTURE(angle);
			CHECK(Math::abs(bend.rest - angle) <= 1e-5);
			if (folded) {
				CHECK(Math::abs(bend.rest) > 0.1);
			} else {
				CHECK(bend.rest == 0.0f);
			}
		}
	}

	TEST_CASE("remaining B06 build counts") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 0, true);
		Dictionary bend;
		bend["compliance"] = 1e-4;
		Dictionary lra;
		lra["type"] = 1;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", bend));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", lra));
		for (int generation = 0; generation < 3; ++generation) {
			CAPTURE(generation);
			const auto before = Probe::build_counts(body.rid());
			bend["compliance"] = 0.002 + generation * 0.001;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", bend));
			const auto after = Probe::build_counts(body.rid());
			CHECK(after.generation == before.generation + 1);
			CHECK(after.create_constraints == before.create_constraints + 1);
			CHECK(after.optimize == before.optimize + 1);
			CHECK(Probe::dihedral_bend_count(body.rid()) == 1);
			CHECK(Probe::lra_count(body.rid()) == 3);
		}
	}

	TEST_CASE("remaining B07 shrink composition") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool bend : { false, true }) {
			for (float shrink : { 0.0f, 0.5f }) {
				CAPTURE(bend);
				CAPTURE(shrink);
				MeshBodyScope body(server.ptr(), space.rid());
				server->soft_body_set_shrinking_factor(body.rid(), shrink);
				if (bend) {
					Dictionary config;
					config["compliance"] = 1e-4;
					REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
				}
				server->soft_body_set_space(body.rid(), RID());
				server->soft_body_set_space(body.rid(), space.rid());
				REQUIRE(Probe::edge_count(body.rid()) == 6);
				for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
					const auto edge = Probe::edge(body.rid(), i);
					const float length = Probe::vertex_position(body.rid(), edge.vertices[0]).distance_to(Probe::vertex_position(body.rid(), edge.vertices[1]));
					CHECK(edge.rest == doctest::Approx(length * (1 - shrink)).epsilon(1e-5));
				}
			}
		}
	}

	TEST_CASE("remaining B08 cantilever") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		auto fixture = fixture_grid();
		// Binary-exact spacing avoids a spurious initial bend angle from the
		// sqrt(dot-square) normalization of nearly equal decimal grid edges.
		for (int z = 0; z < 21; ++z) {
			for (int x = 0; x < 21; ++x) {
				fixture.vertices.set(z * 21 + x, Vector3(float(x - 10) / 16, 0, float(z) / 16));
			}
		}
		float free_y[2];
		for (int mode = 0; mode < 2; ++mode) {
			CAPTURE(mode);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			MeshBodyScope body(server.ptr(), space.rid(), &fixture);
			// Original damping 0.01 leaves a large oscillation at the frozen 120-step
			// endpoint. Shared damping 5/s dissipates transients (0.2 s timescale).
			server->soft_body_set_damping_coefficient(body.rid(), 5.0f);
			Dictionary config;
			config["compliance"] = mode == 0 ? 0.0 : 1e-3;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
			REQUIRE(Probe::dihedral_bend_count(body.rid()) > 0);
			for (int i = 0; i < Probe::dihedral_bend_count(body.rid()); ++i) {
				const auto bend = Probe::dihedral(body.rid(), i);
				REQUIRE(bend.compliance == doctest::Approx(mode == 0 ? 0.0 : 1e-3).epsilon(1e-5));
				REQUIRE(bend.rest == 0.0f);
			}
			for (int i = 0; i < Probe::vertex_count(body.rid()); ++i) {
				const Vector3 position = Probe::world_position(body.rid(), i);
				const bool pinned = position.z < 0.075f;
				REQUIRE(Probe::shared_inv_mass(body.rid(), i) == (pinned ? 0.0f : 441.0f));
				REQUIRE(Probe::vertex_inv_mass(body.rid(), i) == (pinned ? 0.0f : 441.0f));
			}
			for (int frame = 1; frame <= 120; ++frame) {
				space.step();
				if (frame % 20 == 0) {
					float mean_y = 0;
					float speed = 0;
					for (int source = 420; source < 441; ++source) {
						mean_y += server->soft_body_get_point_global_position(body.rid(), source).y / 21;
					}
					for (int i = 0; i < Probe::vertex_count(body.rid()); ++i) {
						speed += Probe::velocity(body.rid(), i).length_squared() / 441;
					}
					print_line(vformat("B08 qualification mode=%d frame=%d mean_y=%s mean_speed_sq=%s", mode, frame, mean_y, speed));
				}
			}
			REQUIRE(Probe::all_finite(body.rid()));
			free_y[mode] = 0;
			for (int source = 420; source < 441; ++source) {
				const Vector3 position = server->soft_body_get_point_global_position(body.rid(), source);
				free_y[mode] += position.y / 21;
			}
			CHECK(Math::abs(free_y[mode]) > 1e-3f);
			for (int pin : fixture.pins) {
				CHECK(server->soft_body_get_point_global_position(body.rid(), pin).distance_to(fixture.vertices[pin]) <= 1e-4f);
			}
		}
		CAPTURE(free_y[0]);
		CAPTURE(free_y[1]);
		CHECK(free_y[0] > free_y[1] + 1e-3f);
	}

	TEST_CASE("remaining B09 bend pin gate") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int pins = 0; pins <= 2; ++pins) {
			CAPTURE(pins);
			MeshBodyScope body(server.ptr(), space.rid());
			if (pins >= 1) {
				server->soft_body_pin_point(body.rid(), 1, true);
			}
			if (pins == 2) {
				server->soft_body_pin_point(body.rid(), 3, true);
			}
			Dictionary config;
			config["compliance"] = 0.0;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
			CHECK(Probe::dihedral_bend_count(body.rid()) == (pins == 2 ? 0 : 1));
		}
	}

	TEST_CASE("remaining B10 bend invalid") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary good;
		good["compliance"] = 1e-4;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", good));
		for (int kind = 0; kind < 7; ++kind) {
			CAPTURE(kind);
			Dictionary bad = good.duplicate(true);
			if (kind < 3) {
				const int types[] = { -1, 2, INT32_MAX };
				bad["type"] = types[kind];
			} else if (kind == 3) {
				bad.erase("compliance");
			} else {
				const double values[] = { NAN, -1.0, double(FLT_MAX) * 2 };
				bad["compliance"] = values[kind - 4];
			}
			const uint64_t identity = Probe::body_identity(body.rid());
			const auto counts = Probe::build_counts(body.rid());
			const Vector3 position = Probe::world_position(body.rid(), 0);
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "bend/config", bad));
			CHECK(errors.has(kind == 3 ? "SBREM-KEY" : "SBREM-VALUE"));
			CHECK(errors.has("bend/config"));
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "bend/config")) == good);
			CHECK(Probe::body_identity(body.rid()) == identity);
			CHECK(Probe::world_position(body.rid(), 0) == position);
			CHECK(Probe::build_counts(body.rid()).generation == counts.generation);
		}
	}

	TEST_CASE("remaining L01 no anchor") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		auto fixture = fixture_graph();
		fixture.pins.clear();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		Dictionary config;
		config["type"] = 1;
		ErrorCapture errors;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
		CHECK(errors.count() == 1);
		CHECK(errors.has("SBREM-LRA-NO-ANCHOR"));
		CHECK(errors.has("skipped 12"));
		CHECK(Probe::lra_count(body.rid()) == 0);
		errors.clear();
		space.step(30);
		CHECK(errors.count() == 0);
		server->soft_body_set_space(body.rid(), RID());
		server->soft_body_set_space(body.rid(), space.rid());
		CHECK(errors.count() == 1);
		CHECK(errors.has("SBREM-LRA-NO-ANCHOR"));
		CHECK(Probe::lra_count(body.rid()) == 0);
	}

	TEST_CASE("remaining L02 anchored graph") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_graph();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		Dictionary config;
		config["type"] = 1;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
		REQUIRE(Probe::lra_count(body.rid()) == 11);
		CHECK(int(server->soft_body_get_extra_property(body.rid(), "lra/count")) == 11);
		HashSet<int> targets;
		for (int i = 0; i < 11; ++i) {
			const auto constraint = Probe::lra(body.rid(), i);
			CHECK(Probe::shared_inv_mass(body.rid(), constraint.vertices[0]) == 0);
			CHECK(Probe::shared_inv_mass(body.rid(), constraint.vertices[1]) > 0);
			CHECK_FALSE(targets.has(constraint.vertices[1]));
			targets.insert(constraint.vertices[1]);
		}
	}

	TEST_CASE("remaining L03 pin rebuild") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_graph();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		Dictionary config;
		config["type"] = 2;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
		for (int operation = 0; operation < 4; ++operation) {
			CAPTURE(operation);
			const auto before = Probe::build_counts(body.rid());
			if (operation == 1) {
				server->soft_body_set_total_mass(body.rid(), 2.0);
			} else if (operation != 0) {
				server->soft_body_pin_point(body.rid(), 11, operation == 3);
			} else {
				server->soft_body_pin_point(body.rid(), 11, true);
			}
			CHECK(Probe::build_counts(body.rid()).generation == before.generation + 1);
			CHECK(Probe::lra_count(body.rid()) == (operation == 2 ? 11 : 10));
			for (int i = 0; i < Probe::lra_count(body.rid()); ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				CHECK(Probe::vertex_inv_mass(body.rid(), constraint.vertices[0]) == 0);
				CHECK(Probe::vertex_inv_mass(body.rid(), constraint.vertices[1]) > 0);
			}
		}
	}

	TEST_CASE("remaining L04 distance modes") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_graph();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		const int count = Probe::vertex_count(body.rid());
		Vector<double> distances;
		distances.resize(count);
		distances.fill(1e100);
		distances.set(0, 0);
		// Independent repeated edge relaxation finds graph shortest distances.
		for (int pass = 0; pass < count; ++pass) {
			for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
				const auto edge = Probe::edge(body.rid(), i);
				const int a = edge.vertices[0], b = edge.vertices[1];
				distances.set(a, MIN(distances[a], distances[b] + edge.rest));
				distances.set(b, MIN(distances[b], distances[a] + edge.rest));
			}
		}
		bool separated = false;
		for (int type : { 1, 2 }) {
			CAPTURE(type);
			Dictionary config;
			config["type"] = type;
			config["max_distance_multiplier"] = 1.2;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			REQUIRE(Probe::lra_count(body.rid()) == count - 1);
			for (int i = 0; i < count - 1; ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				CHECK(constraint.vertices[0] == 0);
				const int target = constraint.vertices[1];
				const double straight = Probe::vertex_position(body.rid(), 0).distance_to(Probe::vertex_position(body.rid(), target));
				CHECK(constraint.rest == doctest::Approx(1.2 * (type == 1 ? straight : distances[target])).epsilon(1e-5));
				separated |= distances[target] > straight + 1.0;
			}
		}
		CHECK(separated);
	}

	TEST_CASE("remaining L05 disconnected") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool all_pinned : { false, true }) {
			CAPTURE(all_pinned);
			auto fixture = fixture_graph(true);
			if (all_pinned) {
				fixture.pins.clear();
				for (int i = 0; i < fixture.vertices.size(); ++i) {
					fixture.pins.push_back(i);
				}
			}
			MeshBodyScope body(server.ptr(), space.rid(), &fixture);
			Dictionary config;
			config["type"] = 2;
			ErrorCapture errors;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			CHECK(Probe::lra_count(body.rid()) == (all_pinned ? 0 : 11));
			CHECK(errors.count() == (all_pinned ? 0 : 1));
			if (!all_pinned) {
				CHECK(errors.has("SBREM-LRA-NO-ANCHOR"));
				CHECK(errors.has("skipped 3"));
			}
		}
	}

	TEST_CASE("remaining L06 vertex types") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 0, true);
		for (int length : { 3, 4 }) {
			CAPTURE(length);
			PackedInt32Array types;
			for (int i = 0; i < length; ++i) {
				types.push_back(i == 1 ? 0 : (i == 0 ? 1 : 2));
			}
			Dictionary config;
			config["type"] = types;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			REQUIRE(Probe::lra_count(body.rid()) == 2);
			for (int i = 0; i < 2; ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				CHECK(constraint.vertices[1] != 1);
			}
			const auto identity = Probe::body_identity(body.rid());
			types.fill(0);
			config["type"] = types;
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			CHECK(errors.has("SBREM-VALUE"));
			CHECK(errors.has("type"));
			CHECK(Probe::body_identity(body.rid()) == identity);
		}
		const auto graph = fixture_graph();
		MeshBodyScope folded(server.ptr(), space.rid(), &graph);
		const int count = graph.vertices.size();
		Vector<double> distances;
		distances.resize(count);
		distances.fill(1e100);
		distances.set(0, 0);
		// Independent Bellman-Ford over actual stretch graph, never LRA outputs.
		for (int pass = 0; pass < count; ++pass) {
			for (int i = 0; i < Probe::edge_count(folded.rid()); ++i) {
				const auto edge = Probe::edge(folded.rid(), i);
				const int a = edge.vertices[0], b = edge.vertices[1];
				const double length = Probe::vertex_position(folded.rid(), a).distance_to(Probe::vertex_position(folded.rid(), b));
				distances.set(a, MIN(distances[a], distances[b] + length));
				distances.set(b, MIN(distances[b], distances[a] + length));
			}
		}
		for (int length : { count - 1, count }) {
			CAPTURE(length);
			PackedInt32Array types;
			for (int i = 0; i < length; ++i) {
				types.push_back(i % 3);
			}
			Dictionary config;
			config["type"] = types;
			REQUIRE(server->soft_body_set_extra_property(folded.rid(), "lra/config", config));
			HashSet<int> enabled;
			bool separated[3] = {};
			for (int i = 0; i < Probe::lra_count(folded.rid()); ++i) {
				const auto constraint = Probe::lra(folded.rid(), i);
				const int target = constraint.vertices[1];
				const Vector3 point = Probe::vertex_position(folded.rid(), target);
				const int source = graph.vertices.find(point);
				REQUIRE(source >= 0);
				const int type = types[MIN(source, length - 1)];
				CHECK(type != 0);
				CHECK(constraint.vertices[0] == 0);
				CHECK_FALSE(enabled.has(source));
				enabled.insert(source);
				const double straight = point.distance_to(graph.vertices[0]);
				CHECK(constraint.rest == doctest::Approx(type == 1 ? straight : distances[target]).epsilon(1e-5));
				separated[type] |= distances[target] > straight + 1.0;
			}
			for (int source = 1; source < count; ++source) {
				CHECK(enabled.has(source) == (types[MIN(source, length - 1)] != 0));
			}
			CHECK(separated[1]);
			CHECK(separated[2]);
		}
	}

	TEST_CASE("remaining L07 distance scales") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 0, true);
		for (int shape = 0; shape < 4; ++shape) {
			CAPTURE(shape);
			Dictionary config;
			config["type"] = 1;
			PackedFloat32Array values;
			for (float value : { 0.2f, 1.1f, 1.3f, 1.7f }) {
				values.push_back(value);
			}
			if (shape == 1) {
				config["max_distance_multiplier"] = 1.2;
			} else if (shape >= 2) {
				values.resize(shape == 2 ? 4 : 3);
				config["max_distance_multiplier"] = values;
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			for (int i = 0; i < 3; ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				const int target = constraint.vertices[1];
				const double multiplier = shape == 0 ? 1 : shape == 1 ? 1.2
																	  : values[MIN(target, values.size() - 1)];
				CHECK(constraint.rest == doctest::Approx(Probe::vertex_position(body.rid(), target).length() * multiplier).epsilon(1e-5));
			}
			const auto id = Probe::body_identity(body.rid());
			for (double value : { 0.0, -1.0, double(NAN), double(INFINITY), double(FLT_MAX) * 2 }) {
				CAPTURE(value);
				Dictionary bad = config.duplicate(true);
				bad["max_distance_multiplier"] = value;
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "lra/config", bad));
				CHECK(errors.has("SBREM-VALUE"));
				CHECK(errors.has("max_distance_multiplier"));
				CHECK(Probe::body_identity(body.rid()) == id);
			}
		}
	}

	TEST_CASE("remaining L09 stretch bound") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		const auto fixture = fixture_graph();
		float excess[2] = {};
		for (bool enabled : { false, true }) {
			CAPTURE(enabled);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			MeshBodyScope body(server.ptr(), space.rid(), &fixture);
			server->soft_body_set_linear_stiffness(body.rid(), 0.05);
			Dictionary config;
			config["type"] = 1;
			// Capture exact production constraints for both controls, then clear only
			// LRA for disabled control; edge material and anchor remain identical.
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			Vector<Probe::Constraint> limits;
			for (int i = 0; i < Probe::lra_count(body.rid()); ++i) {
				limits.push_back(Probe::lra(body.rid(), i));
			}
			REQUIRE(limits.size() == 11);
			if (!enabled) {
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", Variant()));
			}
			space.step(180);
			REQUIRE(Probe::all_finite(body.rid()));
			for (const auto &limit : limits) {
				const float length = Probe::world_position(body.rid(), limit.vertices[0]).distance_to(Probe::world_position(body.rid(), limit.vertices[1]));
				excess[enabled] = MAX(excess[enabled], length - limit.rest);
				if (enabled) {
					CHECK(length <= limit.rest * 1.01f);
				}
			}
		}
		CAPTURE(excess[0]);
		CAPTURE(excess[1]);
		CHECK(excess[0] > 1e-3f);
	}

	TEST_CASE("remaining L10 edge independence") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (float shrink : { 0.0f, 0.5f }) {
			CAPTURE(shrink);
			const auto fixture = fixture_graph();
			MeshBodyScope body(server.ptr(), space.rid(), &fixture);
			server->soft_body_set_shrinking_factor(body.rid(), shrink);
			// Both controls use the new mesh builder and identical typed base.
			Dictionary bend;
			bend["compliance"] = double(FLT_MAX);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", bend));
			Vector<Probe::Constraint> before;
			for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
				before.push_back(Probe::edge(body.rid(), i));
			}
			Dictionary config;
			config["type"] = 2;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
			CHECK(Probe::edge_count(body.rid()) == before.size());
			for (const auto &edge : before) {
				int matches = 0;
				for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
					const auto other = Probe::edge(body.rid(), i);
					if (edge.vertices[0] == other.vertices[0] && edge.vertices[1] == other.vertices[1]) {
						++matches;
						CHECK(edge.rest == other.rest);
						CHECK(edge.compliance == other.compliance);
					}
				}
				CHECK(matches == 1);
			}
		}
	}

	TEST_CASE("remaining L11 pin mapping") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_seam();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		server->soft_body_pin_point(body.rid(), 4, true);
		Dictionary config;
		config["type"] = 1;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
		for (int source : { 4, 5 }) {
			CAPTURE(source);
			if (source == 5) {
				server->soft_body_pin_point(body.rid(), 4, false);
				server->soft_body_pin_point(body.rid(), 5, true);
			}
			REQUIRE(Probe::lra_count(body.rid()) == 3);
			for (int i = 0; i < 3; ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				CHECK(Probe::world_position(body.rid(), constraint.vertices[0]) == fixture.vertices[source]);
			}
		}
	}

	TEST_CASE("remaining L12 type boundaries") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 0, true);
		Dictionary good;
		good["type"] = 1;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", good));
		for (int shape = 0; shape < 3; ++shape) {
			for (int number : { -1, 0, 3 }) {
				CAPTURE(shape);
				CAPTURE(number);
				Dictionary config;
				if (shape == 0) {
					config["type"] = number;
				} else if (shape == 1) {
					PackedInt32Array values;
					values.push_back(number);
					config["type"] = values;
				} else {
					config["type"] = bool(number);
				}
				const auto id = Probe::body_identity(body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "lra/config", config));
				CHECK(errors.has(shape == 2 ? "SBREM-TYPE" : "SBREM-VALUE"));
				CHECK(errors.has("lra/config.type"));
				CHECK(Probe::body_identity(body.rid()) == id);
			}
		}
		good["type"] = 2;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", good));
		CHECK(Probe::lra_count(body.rid()) == 3);
	}

	TEST_CASE("remaining A01 broadcast") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const char *keys[] = { "vertex/config", "vertex/config", "bend/config", "lra/config", "lra/config" };
		const char *fields[] = { "edge_scale", "shear_scale", "compliance", "type", "max_distance_multiplier" };
		for (int field = 0; field < 5; ++field) {
			CAPTURE(fields[field]);
			MeshBodyScope body(server.ptr(), space.rid());
			server->soft_body_pin_point(body.rid(), 0, true);
			Dictionary config;
			if (field >= 3) {
				config["type"] = 1;
			}
			config[fields[field]] = field == 3 ? Variant(2) : Variant(field == 2 ? 1e-4 : 1.2);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], config));
			Vector<Probe::Constraint> edges;
			for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
				edges.push_back(Probe::edge(body.rid(), i));
			}
			const auto bend = Probe::dihedral(body.rid(), 0);
			Vector<Probe::Constraint> lras;
			for (int i = 0; i < Probe::lra_count(body.rid()); ++i) {
				lras.push_back(Probe::lra(body.rid(), i));
			}
			if (field == 3) {
				PackedInt32Array array;
				array.push_back(2);
				config[fields[field]] = array;
			} else {
				config[fields[field]] = make_floats(1, field == 2 ? 1e-4 : 1.2);
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], config));
			CHECK(Probe::edge_count(body.rid()) == edges.size());
			for (int i = 0; i < edges.size(); ++i) {
				const auto other = Probe::edge(body.rid(), i);
				CHECK(other.vertices[0] == edges[i].vertices[0]);
				CHECK(other.vertices[1] == edges[i].vertices[1]);
				CHECK(other.compliance == edges[i].compliance);
				CHECK(other.rest == edges[i].rest);
			}
			if (bend.valid) {
				CHECK(Probe::dihedral(body.rid(), 0).compliance == bend.compliance);
			}
			CHECK(Probe::lra_count(body.rid()) == lras.size());
			for (int i = 0; i < lras.size(); ++i) {
				CHECK(Probe::lra(body.rid(), i).vertices[1] == lras[i].vertices[1]);
				CHECK(Probe::lra(body.rid(), i).rest == lras[i].rest);
			}
		}
	}

	TEST_CASE("remaining A02 tail extension") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const char *keys[] = { "vertex/config", "vertex/config", "bend/config", "lra/config", "lra/config" };
		const char *fields[] = { "edge_scale", "shear_scale", "compliance", "type", "max_distance_multiplier" };
		for (int field = 0; field < 5; ++field) {
			for (int length : { 2, 3 }) {
				CAPTURE(fields[field]);
				CAPTURE(length);
				RemainingMeshFixture fixture;
				// Put the shared diagonal at the last two source indices: short
				// tails must reach both bend and shear, not only peripheral edge columns.
				for (const Vector3 &point : { Vector3(0, 0, 0), Vector3(1, 0, 1), Vector3(1, 0, 0), Vector3(0, 0, 1) }) {
					fixture.vertices.push_back(point);
				}
				for (int index : { 2, 0, 3, 2, 3, 1 }) {
					fixture.indices.push_back(index);
				}
				MeshBodyScope body(server.ptr(), space.rid(), &fixture);
				server->soft_body_pin_point(body.rid(), 0, true);
				Dictionary config;
				if (field >= 3) {
					config["type"] = 1;
				}
				PackedFloat32Array floats;
				PackedInt32Array ints;
				for (int i = 0; i < length; ++i) {
					floats.push_back((i + 1) * (field == 2 ? 1e-4f : 1.0f));
					ints.push_back(i == 0 ? 0 : 2);
				}
				config[fields[field]] = field == 3 ? Variant(ints) : Variant(floats);
				REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], config));
				Vector<Probe::Constraint> short_edges, short_lra;
				for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
					short_edges.push_back(Probe::edge(body.rid(), i));
				}
				for (int i = 0; i < Probe::lra_count(body.rid()); ++i) {
					short_lra.push_back(Probe::lra(body.rid(), i));
				}
				const auto short_bend = Probe::dihedral(body.rid(), 0);
				while (floats.size() < 4) {
					floats.push_back(floats[floats.size() - 1]);
				}
				while (ints.size() < 4) {
					ints.push_back(ints[ints.size() - 1]);
				}
				config[fields[field]] = field == 3 ? Variant(ints) : Variant(floats);
				REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], config));
				CHECK(Probe::edge_count(body.rid()) == short_edges.size());
				for (int i = 0; i < short_edges.size(); ++i) {
					CHECK(Probe::edge(body.rid(), i).compliance == short_edges[i].compliance);
				}
				CHECK(Probe::lra_count(body.rid()) == short_lra.size());
				for (int i = 0; i < short_lra.size(); ++i) {
					CHECK(Probe::lra(body.rid(), i).rest == short_lra[i].rest);
				}
				if (short_bend.valid) {
					CHECK(Probe::dihedral(body.rid(), 0).compliance == short_bend.compliance);
				}
			}
		}
	}

	TEST_CASE("remaining A03 spatial distribution") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config;
		PackedFloat32Array edge_scales, shear_scales;
		for (float value : { 1.0f, 2.0f, 4.0f, 8.0f }) {
			edge_scales.push_back(value);
		}
		for (float value : { 3.0f, 5.0f, 7.0f, 9.0f }) {
			shear_scales.push_back(value);
		}
		config["edge_scale"] = edge_scales;
		config["shear_scale"] = shear_scales;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", config));
		const double base = (1.0 / 60 / 20) * (1.0 / 60 / 20) * 8;
		REQUIRE(Probe::edge_count(body.rid()) == 6);
		for (int i = 0; i < 6; ++i) {
			const auto edge = Probe::edge(body.rid(), i);
			const int a = edge.vertices[0], b = edge.vertices[1];
			const bool shear = (a + b == 2 && MIN(a, b) == 0) || (a + b == 4 && MIN(a, b) == 1);
			const auto &scales = shear ? shear_scales : edge_scales;
			CHECK(edge.compliance == doctest::Approx(base * (scales[a] + scales[b]) / 2).epsilon(1e-5));
		}
		Dictionary bend;
		PackedFloat32Array compliance;
		for (float value : { 0.001f, 0.003f, 0.007f, 0.011f }) {
			compliance.push_back(value);
		}
		bend["compliance"] = compliance;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", bend));
		REQUIRE(Probe::dihedral_bend_count(body.rid()) == 1);
		const auto constraint = Probe::dihedral(body.rid(), 0);
		CHECK(constraint.compliance == doctest::Approx((compliance[constraint.vertices[0]] + compliance[constraint.vertices[1]]) / 2.0).epsilon(1e-5));
	}

	TEST_CASE("remaining A04 array bounds") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const char *keys[] = { "vertex/config", "vertex/config", "bend/config", "lra/config", "lra/config" };
		const char *fields[] = { "edge_scale", "shear_scale", "compliance", "type", "max_distance_multiplier" };
		for (int field = 0; field < 5; ++field) {
			CAPTURE(fields[field]);
			MeshBodyScope body(server.ptr(), space.rid());
			server->soft_body_pin_point(body.rid(), 0, true);
			Dictionary good;
			if (field >= 3) {
				good["type"] = 1;
			}
			good[fields[field]] = field == 3 ? Variant(1) : Variant(1.0);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], good));
			for (int size : { 0, 5, 65537 }) {
				CAPTURE(size);
				Dictionary bad = good.duplicate(true);
				if (field == 3) {
					PackedInt32Array values;
					values.resize(size);
					values.fill(1);
					bad[fields[field]] = values;
				} else {
					bad[fields[field]] = make_floats(size, 1);
				}
				const auto id = Probe::body_identity(body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), keys[field], bad));
				CHECK(errors.has("SBREM-SIZE"));
				CHECK(errors.has(fields[field]));
				CHECK(Probe::body_identity(body.rid()) == id);
				CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), keys[field])) == good);
			}
		}
	}

	TEST_CASE("remaining A05 welded equal") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		auto fixture = fixture_seam();
		fixture.indices.clear();
		for (int index : { 4, 5, 3, 0, 1, 2 }) {
			fixture.indices.push_back(index);
		}
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		for (int shape = 0; shape < 3; ++shape) {
			CAPTURE(shape);
			Dictionary config;
			PackedFloat32Array values = make_floats(7, 2);
			if (shape == 2) {
				for (int i = 0; i < 4; ++i) {
					values.set(i, float(1 << i));
				}
				values.set(4, values[0]);
				values.set(5, values[2]);
			}
			config["edge_scale"] = shape > 0 ? Variant(values) : Variant(2.0);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", config));
			CHECK(Probe::vertex_count(body.rid()) == 4);
			REQUIRE(Probe::edge_count(body.rid()) == 5);
			const double base = (1.0 / 60 / 20) * (1.0 / 60 / 20) * 8;
			for (int i = 0; i < 5; ++i) {
				const auto edge = Probe::edge(body.rid(), i);
				double scales[2] = {};
				for (int end = 0; end < 2; ++end) {
					for (int source = 0; source < 4; ++source) {
						if (Probe::world_position(body.rid(), edge.vertices[end]) == fixture.vertices[source]) {
							scales[end] = values[source];
						}
					}
					REQUIRE(scales[end] > 0);
				}
				CHECK(edge.compliance == doctest::Approx(base * (scales[0] + scales[1]) / 2).epsilon(1e-5));
			}
		}
	}

	TEST_CASE("remaining A06 welded conflict") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const char *keys[] = { "vertex/config", "vertex/config", "bend/config", "lra/config", "lra/config" };
		const char *fields[] = { "edge_scale", "shear_scale", "compliance", "type", "max_distance_multiplier" };
		for (int field = 0; field < 5; ++field) {
			for (float stiffness : { 0.5f, 1.0f }) {
				CAPTURE(fields[field]);
				CAPTURE(stiffness);
				const auto fixture = fixture_seam();
				MeshBodyScope body(server.ptr(), space.rid(), &fixture);
				server->soft_body_pin_point(body.rid(), 1, true);
				server->soft_body_set_linear_stiffness(body.rid(), stiffness);
				Dictionary good;
				if (field >= 3) {
					good["type"] = 1;
				}
				good[fields[field]] = field == 3 ? Variant(1) : Variant(1.0);
				REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[field], good));
				Dictionary bad = good.duplicate(true);
				if (field == 3) {
					PackedInt32Array values;
					values.resize(7);
					values.fill(1);
					values.set(4, 2);
					bad[fields[field]] = values;
				} else {
					PackedFloat32Array values = make_floats(7, 1);
					values.set(4, 2);
					bad[fields[field]] = values;
				}
				const auto id = Probe::body_identity(body.rid());
				const auto generation = Probe::build_counts(body.rid()).generation;
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), keys[field], bad));
				CHECK(errors.has("SBREM-ALIAS"));
				CHECK(errors.has("0 and 4"));
				CHECK(Probe::body_identity(body.rid()) == id);
				CHECK(Probe::build_counts(body.rid()).generation == generation);
				CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), keys[field])) == good);
				if (stiffness == 1.0f) {
					errors.clear();
					server->soft_body_set_linear_stiffness(body.rid(), 0.5f);
					CHECK(errors.count() == 0);
					CHECK(Probe::build_counts(body.rid()).generation == generation + 1);
				}
			}
		}
	}

	TEST_CASE("remaining A08 stiffness product") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int cap = 0; cap < 4; ++cap) {
			for (double scale : { 0.25, 1.0, 4.0 }) {
				CAPTURE(cap);
				CAPTURE(scale);
				MeshBodyScope body(server.ptr(), space.rid());
				server->soft_body_pin_point(body.rid(), 0, true);
				Dictionary config;
				if (cap == 0) {
					config["edge_scale"] = scale;
					config["shear_scale"] = scale;
				}
				if (cap == 1) {
					config["compliance"] = 1e-4;
				}
				if (cap == 2) {
					config["type"] = 1;
				}
				const char *keys[] = { "vertex/config", "bend/config", "lra/config" };
				if (cap < 3) {
					REQUIRE(server->soft_body_set_extra_property(body.rid(), keys[cap], config));
				}
				for (int change = 0; change < 4; ++change) {
					CAPTURE(change);
					server->soft_body_set_total_mass(body.rid(), 1.0);
					server->soft_body_set_simulation_precision(body.rid(), 20);
					server->soft_body_set_linear_stiffness(body.rid(), 0.5);
					const auto before = Probe::build_counts(body.rid());
					const float old_compliance = Probe::edge(body.rid(), 0).compliance;
					if (change == 0) {
						server->soft_body_set_total_mass(body.rid(), 2.0);
					}
					if (change == 1) {
						server->soft_body_set_simulation_precision(body.rid(), 40);
					}
					if (change == 2) {
						server->soft_body_set_linear_stiffness(body.rid(), 0.25);
					}
					if (change == 3) {
						server->soft_body_set_linear_stiffness(body.rid(), 1.0);
					}
					CHECK(Probe::build_counts(body.rid()).generation == before.generation + (cap < 3 ? 1 : 0));
					if (cap == 3) {
						CHECK(Probe::edge(body.rid(), 0).compliance == old_compliance);
					} else {
						const double factor[] = { 0.5, 0.25, 3.0, 0.0 };
						for (int i = 0; i < Probe::edge_count(body.rid()); ++i) {
							const auto edge = Probe::edge(body.rid(), i);
							const double base = (1.0 / 60 / 20) * (1.0 / 60 / 20) * 8;
							CHECK(edge.compliance == doctest::Approx(base * (cap == 0 ? scale : 1) * factor[change]).epsilon(1e-5));
						}
						if (cap == 1) {
							CHECK(Probe::dihedral(body.rid(), 0).compliance == doctest::Approx(1e-4).epsilon(1e-5));
						}
						if (cap == 2) {
							CHECK(Probe::lra_count(body.rid()) == 3);
						}
					}
				}
			}
		}
	}

	TEST_CASE("remaining A10 column ownership") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 0, true);
		Dictionary vertex, bend, lra;
		vertex["edge_scale"] = 2.0;
		vertex["shear_scale"] = 3.0;
		bend["compliance"] = 0.012;
		lra["type"] = 1;
		lra["max_distance_multiplier"] = 1.5;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", vertex));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", bend));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", lra));
		const double base = (1.0 / 60 / 20) * (1.0 / 60 / 20) * 8;
		for (int i = 0; i < 6; ++i) {
			const auto edge = Probe::edge(body.rid(), i);
			const int a = MIN(edge.vertices[0], edge.vertices[1]), b = MAX(edge.vertices[0], edge.vertices[1]);
			const bool shear = (a == 0 && b == 2) || (a == 1 && b == 3);
			CHECK(edge.compliance == doctest::Approx(base * (shear ? 3 : 2)).epsilon(1e-5));
		}
		CHECK(Probe::dihedral(body.rid(), 0).compliance == doctest::Approx(0.012).epsilon(1e-5));
		for (int i = 0; i < 3; ++i) {
			const auto constraint = Probe::lra(body.rid(), i);
			CHECK(constraint.rest == doctest::Approx(Probe::vertex_position(body.rid(), constraint.vertices[1]).length() * 1.5).epsilon(1e-5));
		}
	}

	TEST_CASE("remaining A11 heterogeneous response") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
		const auto fixture = fixture_grid();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		PackedFloat32Array scales;
		for (const Vector3 &vertex : fixture.vertices) {
			scales.push_back(vertex.x < 0 ? 0.1f : 10.0f);
		}
		Dictionary config;
		config["edge_scale"] = scales;
		config["shear_scale"] = scales;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", config));
		space.step(120);
		REQUIRE(Probe::all_finite(body.rid()));
		float left = 0, right = 0;
		for (int x = 0; x < 10; ++x) {
			left += server->soft_body_get_point_global_position(body.rid(), 420 + x).y / 10;
			right += server->soft_body_get_point_global_position(body.rid(), 431 + x).y / 10;
		}
		CAPTURE(left);
		CAPTURE(right);
		CHECK(Math::abs(left - right) > 1e-3f);
		CHECK(Math::abs(left) > 1e-3f);
		CHECK(Math::abs(right) > 1e-3f);
		for (int pin : fixture.pins) {
			CHECK(server->soft_body_get_point_global_position(body.rid(), pin).distance_to(fixture.vertices[pin]) <= 1e-4f);
		}
	}

	TEST_CASE("remaining A12 resource bounds") {
		using namespace SoftBodyCapValidation;
		for (bool faces : { false, true }) {
			const uint64_t maximum = faces ? 262144 * 3 : 65536;
			CAPTURE(faces);
			String error;
			CHECK(resource_size(maximum, faces ? 3 : 1, maximum, 1, &error, faces ? "mesh.indices" : "mesh.vertices"));
			CHECK_FALSE(resource_size(maximum + 1, faces ? 3 : 1, maximum, 1, &error, faces ? "mesh.indices" : "mesh.vertices"));
			CHECK(error.contains("SBREM-SIZE"));
			CHECK_FALSE(resource_size(UINT64_MAX, 1, UINT64_MAX, 4, &error, "mesh.stride"));
			CHECK(error.contains("SBREM-SIZE"));
		}
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config;
		config["edge_scale"] = 1.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", config));
		CHECK(Probe::vertex_count(body.rid()) == 4);
		CHECK(Probe::face_count(body.rid()) == 2);
	}

	TEST_CASE("remaining V01 minimal tetra") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		CHECK(Probe::vertex_count(body.rid()) == 4);
		CHECK(Probe::volume_count(body.rid()) == 1);
		CHECK(Probe::face_count(body.rid()) == 4);
		CHECK(Probe::edge_count(body.rid()) == 0);
		CHECK(Probe::tetra(body.rid(), 0).rest == doctest::Approx(1));
		CHECK(Probe::tetra(body.rid(), 0).compliance == 0);
		for (int i = 0; i < 4; ++i) {
			CHECK(Probe::shared_inv_mass(body.rid(), i) == 4);
			CHECK(Probe::vertex_inv_mass(body.rid(), i) == 4);
		}
		CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(1.0 / 6).epsilon(1e-5));
		CHECK(Probe::build_counts(body.rid()).create_constraints == 0);
		CHECK(Probe::build_counts(body.rid()).optimize == 1);
	}

	TEST_CASE("remaining V02 boundary extraction") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		Dictionary config = fixture_tetra(true);
		VolumeBodyScope body(server.ptr(), space.rid(), config);
		const PackedInt32Array faces = server->soft_body_get_extra_property(body.rid(), "volume/faces");
		REQUIRE(faces.size() == 18);
		CHECK(Probe::volume_count(body.rid()) == 2);
		for (int i = 0; i < faces.size(); i += 3) {
			CHECK_FALSE((faces[i] < 3 && faces[i + 1] < 3 && faces[i + 2] < 3));
		}
		PackedInt32Array tetrahedra = config["tetrahedra"];
		for (int i = 0; i < 4; ++i) {
			const int temp = tetrahedra[i];
			tetrahedra.set(i, tetrahedra[i + 4]);
			tetrahedra.set(i + 4, temp);
		}
		config["tetrahedra"] = tetrahedra;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", config));
		CHECK(PackedInt32Array(server->soft_body_get_extra_property(body.rid(), "volume/faces")) == faces);
		CHECK(Probe::volume_count(body.rid()) == 2);
	}

	TEST_CASE("remaining V03 winding") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		Vector<Vector3> trajectory;
		for (bool reverse : { false, true }) {
			CAPTURE(reverse);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
			Dictionary config = fixture_tetra();
			PackedInt32Array fixed;
			for (int i : { 0, 1, 2 }) {
				fixed.push_back(i);
			}
			config["fixed"] = fixed;
			PackedInt32Array authored = config["tetrahedra"];
			if (reverse) {
				authored.set(1, 2);
				authored.set(2, 1);
			}
			config["tetrahedra"] = authored;
			VolumeBodyScope body(server.ptr(), space.rid(), config);
			CHECK(PackedInt32Array(Dictionary(server->soft_body_get_extra_property(body.rid(), "volume/config"))["tetrahedra"]) == authored);
			const auto tetra = Probe::tetra(body.rid(), 0);
			const Vector3 a = Probe::vertex_position(body.rid(), tetra.vertices[0]);
			const Vector3 b = Probe::vertex_position(body.rid(), tetra.vertices[1]);
			const Vector3 c = Probe::vertex_position(body.rid(), tetra.vertices[2]);
			const Vector3 d = Probe::vertex_position(body.rid(), tetra.vertices[3]);
			CHECK((b - a).cross(c - a).dot(d - a) > 0);
			const PackedInt32Array faces = server->soft_body_get_extra_property(body.rid(), "volume/faces");
			REQUIRE(faces.size() == 12);
			for (int i = 0; i < faces.size(); i += 3) {
				const Vector3 x = Probe::vertex_position(body.rid(), faces[i]);
				const Vector3 y = Probe::vertex_position(body.rid(), faces[i + 1]);
				const Vector3 z = Probe::vertex_position(body.rid(), faces[i + 2]);
				const int opposite = 6 - faces[i] - faces[i + 1] - faces[i + 2];
				CHECK((z - x).cross(y - x).dot(Probe::vertex_position(body.rid(), opposite) - x) < 0);
			}
			for (int i = 0; i < Probe::face_count(body.rid()); ++i) {
				const auto face = Probe::face(body.rid(), i);
				REQUIRE(face.valid);
				const Vector3 x = Probe::vertex_position(body.rid(), face.vertices[0]);
				const Vector3 y = Probe::vertex_position(body.rid(), face.vertices[1]);
				const Vector3 z = Probe::vertex_position(body.rid(), face.vertices[2]);
				const int opposite = 6 - face.vertices[0] - face.vertices[1] - face.vertices[2];
				CHECK((y - x).cross(z - x).dot(Probe::vertex_position(body.rid(), opposite) - x) < 0);
			}
			REQUIRE(Probe::set_vertex_local(body.rid(), 3, Vector3(0, 0, 0.8f)));
			for (int frame = 0; frame < 30; ++frame) {
				space.step();
				const Vector3 position = Probe::world_position(body.rid(), 3);
				CHECK(Probe::all_finite(body.rid()));
				if (frame == 0) {
					CHECK(position.z > 0.8f);
				}
				if (!reverse) {
					trajectory.push_back(position);
				} else {
					CHECK(position.distance_to(trajectory[frame]) <= 1e-5f);
				}
				CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(position.z / 6).epsilon(1e-5));
			}
		}
	}

	TEST_CASE("remaining V04 topology sizes") {
		using namespace SoftBodyCapValidation;
		String error;
		for (int count : { 4, 65536 }) {
			CHECK(resource_size(count, 4, 65536, 1, &error, "volume/config.vertices"));
		}
		for (int count : { 0, 1, 3, 65537 }) {
			CHECK_FALSE(resource_size(count, 4, 65536, 1, &error, "volume/config.vertices"));
			CHECK(error.contains("SBREM-SIZE"));
		}
		CHECK(resource_size(262144 * 4, 4, 262144 * 4, 1, &error, "volume/config.tetrahedra"));
		CHECK_FALSE(resource_size(262144 * 4 + 4, 4, 262144 * 4, 1, &error, "volume/config.tetrahedra"));
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		for (int count : { 0, 1, 3, 5 }) {
			CAPTURE(count);
			Dictionary bad = fixture_tetra();
			PackedInt32Array indices;
			indices.resize(count);
			indices.fill(0);
			bad["tetrahedra"] = indices;
			const auto id = Probe::body_identity(body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
			CHECK(errors.has("SBREM-SIZE"));
			CHECK(Probe::body_identity(body.rid()) == id);
		}
	}

	TEST_CASE("remaining V05 index validation") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		for (int kind = 0; kind < 6; ++kind) {
			CAPTURE(kind);
			Dictionary bad = fixture_tetra();
			if (kind < 3) {
				PackedInt32Array tetra = bad["tetrahedra"];
				tetra.set(3, kind == 0 ? -1 : kind == 1 ? 4
														: 2);
				bad["tetrahedra"] = tetra;
			} else if (kind == 3) {
				PackedVector3Array vertices = bad["vertices"];
				vertices.push_back(Vector3(2, 2, 2));
				bad["vertices"] = vertices;
			} else {
				PackedInt32Array fixed;
				fixed.push_back(kind == 4 ? -1 : 0);
				fixed.push_back(kind == 4 ? 0 : 0);
				bad["fixed"] = fixed;
			}
			const auto id = Probe::body_identity(body.rid());
			const auto generation = Probe::build_counts(body.rid()).generation;
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
			CHECK(errors.has("SBREM-INDEX"));
			CHECK(errors.has("["));
			CHECK(Probe::body_identity(body.rid()) == id);
			CHECK(Probe::build_counts(body.rid()).generation == generation);
		}
	}

	TEST_CASE("remaining V06 degeneracy") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		const double threshold = 1e-9 * 2.0 * Math::sqrt(2.0);
		CHECK_FALSE(SoftBodyCapValidation::volume_determinant_valid(threshold, 2));
		CHECK_FALSE(SoftBodyCapValidation::volume_determinant_valid(-threshold, 2));
		CHECK(SoftBodyCapValidation::volume_determinant_valid(std::nextafter(threshold, INFINITY), 2));
		CHECK_FALSE(SoftBodyCapValidation::volume_determinant_valid(1e-18, 0));
		CHECK(SoftBodyCapValidation::volume_determinant_valid(std::nextafter(1e-18, INFINITY), 0));
		for (int kind = 0; kind < 5; ++kind) {
			CAPTURE(kind);
			Dictionary config = fixture_tetra();
			PackedVector3Array vertices = config["vertices"];
			if (kind == 0) {
				vertices.set(3, Vector3(0.2f, 0.2f, 0));
			}
			if (kind == 1) {
				vertices.set(3, vertices[0]);
			}
			if (kind == 2) {
				vertices.set(3, Vector3(0, 0, std::nextafter(float(threshold), 0.0f)));
			}
			if (kind == 3) {
				vertices.set(3, Vector3(0, 0, std::nextafter(float(threshold), INFINITY)));
			}
			if (kind == 4) {
				for (int i = 0; i < vertices.size(); ++i) {
					vertices.set(i, vertices[i] * 1e20f);
				}
			}
			config["vertices"] = vertices;
			const auto id = Probe::body_identity(body.rid());
			ErrorCapture errors;
			const bool accepted = server->soft_body_set_extra_property(body.rid(), "volume/config", config);
			CHECK(accepted == (kind == 3));
			if (kind != 3) {
				CHECK(errors.has(kind == 4 ? "SBREM-VALUE" : "SBREM-TOPOLOGY"));
				CHECK(Probe::body_identity(body.rid()) == id);
			} else {
				CHECK(Probe::tetra(body.rid(), 0).rest > threshold);
			}
		}
	}

	TEST_CASE("remaining V07 topology conflicts") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		for (int kind = 0; kind < 3; ++kind) {
			CAPTURE(kind);
			Dictionary config = fixture_tetra(kind != 0);
			PackedVector3Array vertices = config["vertices"];
			PackedInt32Array tetra = config["tetrahedra"];
			if (kind == 0) {
				for (int index : { 3, 2, 1, 0 }) {
					tetra.push_back(index);
				}
			}
			if (kind == 1) {
				vertices.set(4, Vector3(0, 0, 2));
			}
			if (kind == 2) {
				vertices.push_back(Vector3(0, 0, -2));
				for (int index : { 0, 1, 2, 5 }) {
					tetra.push_back(index);
				}
			}
			config["vertices"] = vertices;
			config["tetrahedra"] = tetra;
			const auto id = Probe::body_identity(body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", config));
			CHECK(errors.has("SBREM-TOPOLOGY"));
			CHECK(errors.has("tetrahedra["));
			CHECK(Probe::body_identity(body.rid()) == id);
		}
	}

	TEST_CASE("remaining V08 volume pins") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool mesh : { false, true }) {
			CAPTURE(mesh);
			for (int fixed_count : { 0, 2, 4 }) {
				CAPTURE(fixed_count);
				Dictionary config = fixture_tetra();
				PackedInt32Array fixed;
				for (int i = 0; i < fixed_count; ++i) {
					fixed.push_back(i);
				}
				config["fixed"] = fixed;
				VolumeBodyScope body(server.ptr(), space.rid(), config);
				RID ignored;
				if (mesh) {
					ignored = make_quad_mesh();
					server->soft_body_set_mesh(body.rid(), ignored);
				}
				server->soft_body_set_total_mass(body.rid(), 2.0);
				for (int i = 0; i < 4; ++i) {
					CHECK(Probe::vertex_inv_mass(body.rid(), i) == (i < fixed_count ? 0 : 2));
					CHECK_FALSE(server->soft_body_is_point_pinned(body.rid(), i));
				}
				const auto identity = Probe::body_identity(body.rid());
				const auto settings = Probe::settings_identity(body.rid());
				const auto generation = Probe::build_counts(body.rid()).generation;
				const PackedVector3Array positions = server->soft_body_get_extra_property(body.rid(), "volume/positions");
				ErrorCapture errors;
				server->soft_body_pin_point(body.rid(), -1, true);
				CHECK(errors.has("SBREM-CONFLICT"));
				errors.clear();
				server->soft_body_pin_point(body.rid(), 0, true);
				CHECK(errors.has("SBREM-CONFLICT"));
				errors.clear();
				server->soft_body_pin_point(body.rid(), 0, false);
				server->soft_body_remove_all_pinned_points(body.rid());
				CHECK(errors.count() == 0);
				server->soft_body_set_total_mass(body.rid(), 0);
				CHECK(errors.has("SBREM-CONTEXT"));
				CHECK(server->soft_body_get_total_mass(body.rid()) == 2.0);
				CHECK(Probe::body_identity(body.rid()) == identity);
				CHECK(Probe::settings_identity(body.rid()) == settings);
				CHECK(Probe::build_counts(body.rid()).generation == generation);
				CHECK(PackedVector3Array(server->soft_body_get_extra_property(body.rid(), "volume/positions")) == positions);
				for (int i = 0; i < 4; ++i) {
					CHECK(Probe::vertex_inv_mass(body.rid(), i) == (i < fixed_count ? 0 : 2));
				}
				if (mesh) {
					server->soft_body_set_mesh(body.rid(), RID());
					RenderingServer::get_singleton()->free_rid(ignored);
				}
			}
		}
		MeshBodyScope cloth(server.ptr(), space.rid());
		server->soft_body_pin_point(cloth.rid(), 0, true);
		const auto id = Probe::body_identity(cloth.rid());
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(cloth.rid(), "volume/config", fixture_tetra()));
		CHECK(errors.has("SBREM-CONFLICT"));
		CHECK(Probe::body_identity(cloth.rid()) == id);
		CHECK(server->soft_body_is_point_pinned(cloth.rid(), 0));
	}

	TEST_CASE("remaining V09 atomic topology") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		const auto before = Probe::build_counts(body.rid()).generation;
		Dictionary next = fixture_tetra(true);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", next));
		CHECK(Probe::build_counts(body.rid()).generation == before + 1);
		CHECK(Probe::vertex_count(body.rid()) == 5);
		CHECK(Probe::volume_count(body.rid()) == 2);
		const auto id = Probe::body_identity(body.rid());
		const PackedVector3Array positions = server->soft_body_get_extra_property(body.rid(), "volume/positions");
		Dictionary bad = next.duplicate(true);
		PackedInt32Array indices = bad["tetrahedra"];
		indices.set(7, 99);
		bad["tetrahedra"] = indices;
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
		CHECK(errors.has("SBREM-INDEX"));
		CHECK(Probe::body_identity(body.rid()) == id);
		CHECK(PackedVector3Array(server->soft_body_get_extra_property(body.rid(), "volume/positions")) == positions);
		CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "volume/config")) == next);
	}

	TEST_CASE("remaining V10 volume readback") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		const double rest = server->soft_body_get_extra_property(body.rid(), "volume/current");
		REQUIRE(Probe::set_vertex_local(body.rid(), 3, Vector3(0, 0, 0.8f)));
		const double compressed = server->soft_body_get_extra_property(body.rid(), "volume/current");
		CHECK(compressed == doctest::Approx(rest * 0.8).epsilon(1e-5));
		CHECK(compressed < rest - 1e-3);
		REQUIRE(Probe::clear_faces(body.rid()));
		CHECK(Probe::face_count(body.rid()) == 0);
		CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == compressed);
		REQUIRE(Probe::set_vertex_local(body.rid(), 3, Vector3(0, 0, -0.5f)));
		CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(1.0 / 12.0).epsilon(1e-5));
		VolumeBodyScope pair(server.ptr(), space.rid(), fixture_tetra(true));
		REQUIRE(Probe::set_vertex_local(pair.rid(), 3, Vector3(0, 0, -0.5f)));
		// First tetra inverted (-0.5 determinant), second stays positive (1).
		CHECK(double(server->soft_body_get_extra_property(pair.rid(), "volume/current")) == doctest::Approx((0.5 + 1.0) / 6.0).epsilon(1e-5));
	}

	TEST_CASE("remaining V11 volume remap") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		Dictionary config;
		PackedVector3Array vertices;
		PackedInt32Array tetra;
		for (int i = 0; i < 100; ++i) {
			const Vector3 origin((99 - i) * 3, 0, 0);
			for (const Vector3 &p : { Vector3(), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) }) {
				vertices.push_back(origin + p);
			}
			for (int j = 0; j < 4; ++j) {
				tetra.push_back(4 * i + j);
			}
		}
		PackedInt32Array reversed;
		for (int i = 99; i >= 0; --i) {
			for (int j = 0; j < 4; ++j) {
				reversed.push_back(4 * i + j);
			}
		}
		config["vertices"] = vertices;
		config["tetrahedra"] = reversed;
		VolumeBodyScope body(server.ptr(), space.rid(), config);
		REQUIRE(Probe::volume_count(body.rid()) == 100);
		HashSet<int> found;
		bool reordered = false;
		CHECK(Probe::update_group_count(body.rid()) > 1);
		for (int i = 0; i < 100; ++i) {
			const auto actual = Probe::tetra(body.rid(), i);
			const int original = actual.vertices[0] / 4;
			reordered |= original != 99 - i;
			CHECK_FALSE(found.has(original));
			found.insert(original);
			for (int j = 0; j < 4; ++j) {
				CHECK(actual.vertices[j] == original * 4 + j);
			}
			CHECK(actual.rest == doctest::Approx(1));
		}
		CHECK(found.size() == 100);
		CHECK(reordered);
		CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(100.0 / 6).epsilon(1e-5));
	}

	TEST_CASE("remaining V12 render readback") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
		const Dictionary config = fixture_tetra();
		const PackedVector3Array vertices = config["vertices"];
		VolumeBodyScope body(server.ptr(), space.rid(), config);
		const Transform3D transform(Basis(Vector3(0, 1, 0), 0.6f), Vector3(3, 4, 5));
		server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, transform);
		PackedVector3Array positions = server->soft_body_get_extra_property(body.rid(), "volume/positions");
		REQUIRE(positions.size() == 4);
		for (int i = 0; i < 4; ++i) {
			CHECK(positions[i].distance_to(transform.xform(vertices[i])) <= 1e-5f);
		}
		server->soft_body_apply_central_impulse(body.rid(), Vector3(1, 0, 0));
		space.step(30);
		positions = server->soft_body_get_extra_property(body.rid(), "volume/positions");
		CHECK(positions[0].x > 3.1f);
		for (int i = 0; i < 4; ++i) {
			CHECK(positions[i].distance_to(Probe::world_position(body.rid(), i)) <= 1e-5f);
		}
		const PackedInt32Array faces = server->soft_body_get_extra_property(body.rid(), "volume/faces");
		REQUIRE(faces.size() == 12);
		for (int index : faces) {
			CHECK((index >= 0 && index < 4));
		}
	}

	TEST_CASE("remaining V13 compression response") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		double strain[3];
		PackedInt32Array common_faces;
		for (int mode = 0; mode < 3; ++mode) {
			CAPTURE(mode);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
			Dictionary config = fixture_tetra();
			PackedInt32Array fixed;
			for (int i : { 0, 1, 2 }) {
				fixed.push_back(i);
			}
			config["fixed"] = fixed;
			config["compliance"] = mode == 1 ? 1e-3 : 0.0;
			VolumeBodyScope body(server.ptr(), space.rid(), config, mode == 2);
			server->soft_body_set_pressure_coefficient(body.rid(), 0.05);
			CHECK(server->soft_body_get_pressure_coefficient(body.rid()) == doctest::Approx(0.05));
			CHECK(Probe::edge_count(body.rid()) == 0);
			CHECK(Probe::volume_count(body.rid()) == (mode == 2 ? 0 : 1));
			CHECK(Probe::build_counts(body.rid()).optimize == 1);
			const PackedInt32Array faces = server->soft_body_get_extra_property(body.rid(), "volume/faces");
			if (mode == 0) {
				common_faces = faces;
			} else {
				CHECK(faces == common_faces);
			}
			for (int i = 0; i < 4; ++i) {
				CHECK(Probe::vertex_inv_mass(body.rid(), i) == (i == 3 ? 4 : 0));
			}
			// Pressure 0.05 produces +0.2Z acceleration on this one dynamic
			// apex. Target 20% unconstrained displacement in 0.5 s: a=2*0.2/0.5^2
			// gives net -1.6, hence common external force -1.8. This remains well
			// short of inversion and gives a measurable compliance response.
			for (int frame = 0; frame < 30; ++frame) {
				server->soft_body_apply_central_force(body.rid(), Vector3(0, 0, -1.8f));
				space.step();
			}
			REQUIRE(Probe::all_finite(body.rid()));
			const Vector3 a = Probe::world_position(body.rid(), 0), b = Probe::world_position(body.rid(), 1), c = Probe::world_position(body.rid(), 2), d = Probe::world_position(body.rid(), 3);
			const double volume = Math::abs((b - a).cross(c - a).dot(d - a)) / 6.0;
			strain[mode] = Math::abs(volume * 6 - 1);
			if (mode == 2) {
				CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == 0);
			} else {
				CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(volume).epsilon(1e-5));
			}
		}
		CAPTURE(strain[0]);
		CAPTURE(strain[1]);
		CAPTURE(strain[2]);
		print_line(vformat("V13 strain hard=%s soft=%s pressure_only=%s", strain[0], strain[1], strain[2]));
		CHECK(strain[1] > strain[0] + 1e-4);
		CHECK(strain[1] > 0);
		CHECK(strain[2] > 0);

		// Audit the actual rendered fixture through production construction, not
		// only the minimal tetra: ordering, signs, masses, pressure and scheduling.
		const Dictionary asset = JSON::parse_string(FileAccess::get_file_as_string(TestUtils::get_data_path("jolt_physics/volume_physics_input.json")));
		REQUIRE(asset.has("bodies"));
		const Dictionary sphere = Array(asset["bodies"])[2];
		const Array points = sphere["vertices"], cells = sphere["tetrahedra"];
		PackedVector3Array vertices;
		PackedInt32Array tetrahedra, fixed;
		for (const Variant &point : points) {
			const Array xyz = point;
			vertices.push_back(Vector3(xyz[0], xyz[1], xyz[2]));
		}
		for (const Variant &cell : cells) {
			for (const Variant &index : Array(cell)) {
				tetrahedra.push_back(int(index));
			}
		}
		for (const Variant &index : Array(sphere["fixed"])) {
			fixed.push_back(int(index));
		}
		for (double compliance : { 0.0, 1e-3 }) {
			CAPTURE(compliance);
			SpaceScope space(server.ptr());
			Dictionary config;
			config["vertices"] = vertices;
			config["tetrahedra"] = tetrahedra;
			config["fixed"] = fixed;
			config["compliance"] = compliance;
			VolumeBodyScope body(server.ptr(), space.rid(), config);
			const Dictionary parameters = sphere["parameters"];
			server->soft_body_set_pressure_coefficient(body.rid(), double(parameters["pressure"]));
			server->soft_body_set_damping_coefficient(body.rid(), double(parameters["damping"]));
			CHECK(Probe::scalars(body.rid()).pressure == 12);
			CHECK(Probe::scalars(body.rid()).damping == doctest::Approx(double(parameters["damping"])));
			CHECK(Probe::scalars(body.rid()).iterations == 20);
			CHECK(Probe::edge_count(body.rid()) == 0);
			CHECK(Probe::volume_count(body.rid()) == cells.size());
			CHECK(Probe::face_count(body.rid()) == Array(sphere["faces"]).size());
			CHECK(Probe::build_counts(body.rid()).create_constraints == 0);
			CHECK(Probe::build_counts(body.rid()).optimize == 1);
			double rest = 0;
			for (int i = 0; i < cells.size(); ++i) {
				const auto cell = Probe::tetra(body.rid(), i);
				REQUIRE(cell.valid);
				const Vector3 a = vertices[cell.vertices[0]], b = vertices[cell.vertices[1]], c = vertices[cell.vertices[2]], d = vertices[cell.vertices[3]];
				const double det = (b - a).cross(c - a).dot(d - a);
				CHECK(det > 0);
				CHECK(cell.rest == doctest::Approx(det).epsilon(1e-5));
				CHECK(cell.compliance == doctest::Approx(compliance));
				rest += det / 6;
			}
			double surface = 0;
			for (int i = 0; i < Probe::face_count(body.rid()); ++i) {
				const auto face = Probe::face(body.rid(), i);
				const Vector3 a = vertices[face.vertices[0]], b = vertices[face.vertices[1]], c = vertices[face.vertices[2]];
				CHECK((b - a).cross(c - a).dot(a - vertices[vertices.size() - 1]) > 0);
				surface += a.cross(b).dot(c) / 6.0;
			}
			CHECK(surface == doctest::Approx(rest).epsilon(1e-5));
			CHECK(double(server->soft_body_get_extra_property(body.rid(), "volume/current")) == doctest::Approx(rest).epsilon(1e-5));
			for (int i = 0; i < vertices.size(); ++i) {
				CHECK(Probe::world_position(body.rid(), i).is_equal_approx(vertices[i]));
				CHECK(Probe::shared_inv_mass(body.rid(), i) == (fixed.has(i) ? 0 : vertices.size()));
				CHECK(Probe::vertex_inv_mass(body.rid(), i) == (fixed.has(i) ? 0 : vertices.size()));
			}
		}
	}

	TEST_CASE("remaining V14 volume scope") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_pin_point(body.rid(), 1, true);
		server->soft_body_remove_all_pinned_points(body.rid());
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", fixture_tetra()));
		CHECK(Probe::edge_count(body.rid()) == 0);
		Dictionary config = fixture_tetra();
		PackedInt32Array fixed;
		fixed.push_back(2);
		config["fixed"] = fixed;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", config));
		CHECK(Probe::vertex_inv_mass(body.rid(), 2) == 0);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", Variant()));
		CHECK(Probe::volume_count(body.rid()) == 0);
		CHECK(Probe::edge_count(body.rid()) > 0);
		CHECK(server->soft_body_get_extra_property(body.rid(), "volume/config").get_type() == Variant::NIL);
		for (int i = 0; i < 4; ++i) {
			CHECK_FALSE(server->soft_body_is_point_pinned(body.rid(), i));
		}
		server->soft_body_pin_point(body.rid(), 3, true);
		CHECK(server->soft_body_is_point_pinned(body.rid(), 3));
		VolumeBodyScope standalone(server.ptr(), space.rid(), fixture_tetra());
		REQUIRE(server->soft_body_set_extra_property(standalone.rid(), "volume/config", Variant()));
		CHECK_FALSE(Probe::in_space(standalone.rid()));
	}

	TEST_CASE("remaining V15 inactive readback") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope cloth(server.ptr(), space.rid());
		const RID rod = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), rod, 4));
		server->soft_body_set_space(rod, space.rid());
		for (RID body : { cloth.rid(), rod }) {
			CHECK(int(server->soft_body_get_extra_property(body, "volume/count")) == 0);
			const Variant current = server->soft_body_get_extra_property(body, "volume/current");
			CHECK(current.get_type() == Variant::FLOAT);
			CHECK(double(current) == 0);
			const Variant positions = server->soft_body_get_extra_property(body, "volume/positions");
			CHECK(positions.get_type() == Variant::PACKED_VECTOR3_ARRAY);
			CHECK(PackedVector3Array(positions).is_empty());
			const Variant faces = server->soft_body_get_extra_property(body, "volume/faces");
			CHECK(faces.get_type() == Variant::PACKED_INT32_ARRAY);
			CHECK(PackedInt32Array(faces).is_empty());
		}
		server->free_rid(rod);
	}

	TEST_CASE("remaining V16 volume values") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		VolumeBodyScope body(server.ptr(), space.rid(), fixture_tetra());
		for (int vertex = 0; vertex < 4; ++vertex) {
			for (int axis = 0; axis < 3; ++axis) {
				for (float value : { NAN, INFINITY, -INFINITY }) {
					CAPTURE(vertex);
					CAPTURE(axis);
					CAPTURE(value);
					Dictionary config = fixture_tetra();
					PackedVector3Array vertices = config["vertices"];
					Vector3 point = vertices[vertex];
					point[axis] = value;
					vertices.set(vertex, point);
					config["vertices"] = vertices;
					const auto id = Probe::body_identity(body.rid());
					ErrorCapture errors;
					CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", config));
					CHECK(errors.has("SBREM-VALUE"));
					CHECK(errors.has(vformat("vertices[%d][%d]", vertex, axis)));
					CHECK(Probe::body_identity(body.rid()) == id);
				}
			}
		}
		for (double value : { -1.0, double(NAN), double(INFINITY), double(FLT_MAX) * 2 }) {
			CAPTURE(value);
			Dictionary config = fixture_tetra();
			config["compliance"] = value;
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", config));
			CHECK(errors.has("SBREM-VALUE"));
			CHECK(errors.has("compliance"));
		}
		Dictionary maximum = fixture_tetra();
		maximum["compliance"] = double(FLT_MAX);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "volume/config", maximum));
		CHECK(Probe::tetra(body.rid(), 0).compliance == FLT_MAX);
		for (const char *field : { "vertices", "tetrahedra", "compliance", "fixed" }) {
			CAPTURE(field);
			Dictionary bad = fixture_tetra();
			bad[field] = true;
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
			CHECK(errors.has("SBREM-TYPE"));
			CHECK(errors.has(field));
		}
	}

	TEST_CASE("remaining K01 skin initialization") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (double distance : { 0.0, 0.05, double(FLT_MAX) }) {
			for (bool inspect_before_solve : { false, true }) {
				CAPTURE(distance);
				CAPTURE(inspect_before_solve);
				SpaceScope space(server.ptr());
				server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
				MeshBodyScope body(server.ptr(), space.rid());
				Dictionary config = fixture_skin();
				config["max_distance"] = distance;
				Array pose;
				pose.push_back(Transform3D(Basis(), Vector3(1, 2, 3)));
				config["initial_pose"] = pose;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
				CHECK(Probe::skinned_count(body.rid()) == 4);
				CHECK(Probe::inv_bind_count(body.rid()) == 1);
				Vector<Vector3> expected;
				for (int i = 0; i < 4; ++i) {
					expected.push_back(Probe::world_position(body.rid(), i) + Vector3(1, 2, 3));
					REQUIRE(Probe::set_vertex_local(body.rid(), i, Probe::vertex_position(body.rid(), i), Vector3(0.1f, 0.2f, 0.3f)));
				}
				if (inspect_before_solve) {
					REQUIRE(Probe::invoke_pre_step(body.rid()));
				} else {
					space.step();
				}
				CHECK(Probe::build_counts(body.rid()).skin_init == 1);
				CHECK(Probe::build_counts(body.rid()).skin_recurring == 0);
				for (int i = 0; i < 4; ++i) {
					const auto target = Probe::skin_target(body.rid(), i);
					REQUIRE(target.valid);
					CHECK(target.previous.is_finite());
					CHECK(target.current.is_finite());
					if (inspect_before_solve) {
						CHECK(target.previous == target.current);
					} else {
						// End-of-step COM recentering shifts current, not previous.
						CHECK(target.previous.distance_to(expected[i]) <= 1e-4f);
						CHECK(Probe::frame(body.rid()).xform(target.current).distance_to(expected[i]) <= 1e-4f);
					}
					CHECK(Probe::velocity(body.rid(), i).length() <= 1e-4f);
					CHECK(Probe::world_position(body.rid(), i).distance_to(expected[i]) <= 1e-4f);
				}
				space.step();
				CHECK(Probe::all_finite(body.rid()));
				CHECK(Probe::build_counts(body.rid()).skin_init == 1);
				CHECK(Probe::build_counts(body.rid()).skin_recurring == 1);
			}
		}
	}

	TEST_CASE("remaining K02 blended weights") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int slots : { 1, 2, 4 }) {
			CAPTURE(slots);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config = fixture_skin(2);
			PackedInt32Array joints;
			PackedFloat32Array weights;
			for (int vertex = 0; vertex < 4; ++vertex) {
				for (int slot = 0; slot < 4; ++slot) {
					joints.push_back(slot % 2);
					weights.push_back(slot < slots ? 1.0f / slots : 0);
				}
			}
			config["joint_indices"] = joints;
			config["joint_weights"] = weights;
			Array pose;
			pose.push_back(Transform3D(Basis(Vector3(0, 1, 0), 0.3f), Vector3(1, 2, 0)));
			pose.push_back(Transform3D(Basis().scaled(Vector3(2, 1, 0.5f)), Vector3(-1, 0, 1)));
			config["initial_pose"] = pose;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
			Vector<Vector3> points;
			for (int i = 0; i < 4; ++i) {
				points.push_back(Probe::vertex_position(body.rid(), i));
			}
			REQUIRE(Probe::invoke_pre_step(body.rid()));
			for (int i = 0; i < Probe::skinned_count(body.rid()); ++i) {
				const auto skin = Probe::skin(body.rid(), i);
				Vector3 expected;
				for (int slot = 0; slot < 4; ++slot) {
					CHECK(skin.joints[slot] == slot % 2);
					CHECK(skin.weights[slot] == weights[slot]);
					expected += Transform3D(pose[slot % 2]).xform(points[skin.vertex]) * weights[slot];
				}
				CHECK(Probe::world_position(body.rid(), skin.vertex).distance_to(expected) <= 1e-4f);
			}
		}
	}

	TEST_CASE("remaining K03 skin shape") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const Dictionary good = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		for (const char *field : { "vertices", "joint_indices", "joint_weights", "inv_bind", "initial_pose" }) {
			CAPTURE(field);
			Dictionary bad = good.duplicate(true);
			bad.erase(field);
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
			CHECK(errors.has("SBREM-KEY"));
			CHECK(errors.has(field));
			before.unchanged();
		}
		for (const char *field : { "joint_indices", "joint_weights", "initial_pose" }) {
			for (int change : { -1, 1 }) {
				CAPTURE(field);
				CAPTURE(change);
				Dictionary bad = good.duplicate(true);
				if (String(field) == "joint_indices") {
					PackedInt32Array values = bad[field];
					values.resize(16 + change);
					bad[field] = values;
				} else if (String(field) == "joint_weights") {
					PackedFloat32Array values = bad[field];
					values.resize(16 + change);
					bad[field] = values;
				} else {
					Array values = bad[field];
					values.resize(1 + change);
					bad[field] = values;
				}
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
				CHECK(errors.has("SBREM-SIZE"));
				before.unchanged();
			}
		}
		for (const char *field : { "vertices", "joint_indices", "joint_weights", "inv_bind" }) {
			for (int length : { 0, 1 }) {
				if (String(field) == "inv_bind" && length == 1) {
					continue; // J=1 positive already constructed above.
				}
				CAPTURE(field);
				CAPTURE(length);
				Dictionary bad = good.duplicate(true);
				if (String(field) == "joint_weights") {
					bad[field] = make_floats(length, 1);
				} else if (String(field) == "inv_bind") {
					bad[field] = Array();
					bad["initial_pose"] = Array();
				} else {
					PackedInt32Array values;
					values.resize(length);
					values.fill(0);
					bad[field] = values;
				}
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
				CHECK(errors.has("SBREM-SIZE"));
				before.unchanged();
			}
		}
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", fixture_skin(1, 1)));
		CHECK(Probe::skinned_count(body.rid()) == 1);
		CHECK(Probe::inv_bind_count(body.rid()) == 1);
	}

	TEST_CASE("remaining K04 weight bounds") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const Dictionary good = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		for (float weight : { -1.0f, 0.0f, NAN, INFINITY, std::nextafter(1.0f + 1e-5f, 1.0f), std::nextafter(1.0f + 1e-5f, INFINITY), std::nextafter(1.0f - 1e-5f, 1.0f), std::nextafter(1.0f - 1e-5f, 0.0f) }) {
			CAPTURE(weight);
			Dictionary config = good.duplicate(true);
			PackedFloat32Array weights = config["joint_weights"];
			weights.set(0, weight);
			config["joint_weights"] = weights;
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			const bool valid = std::isfinite(weight) && std::abs(double(weight) - 1.0) <= 1e-5;
			CHECK(server->soft_body_set_extra_property(body.rid(), "skin/config", config) == valid);
			if (!valid) {
				CHECK(errors.has("SBREM-VALUE"));
				CHECK(errors.has("joint_weights["));
				before.unchanged();
			} else {
				CHECK(PackedFloat32Array(Dictionary(server->soft_body_get_extra_property(body.rid(), "skin/config"))["joint_weights"])[0] == weight);
			}
		}
		Dictionary blended = fixture_skin(2, 1);
		blended["max_distance"] = 0.0;
		PackedFloat32Array weights = blended["joint_weights"];
		weights.set(0, 0.25f);
		weights.set(1, 0.750005f);
		blended["joint_weights"] = weights;
		Array poses;
		poses.push_back(Transform3D(Basis(), Vector3(128, 0, 0)));
		poses.push_back(Transform3D(Basis(), Vector3(256, 0, 0)));
		blended["initial_pose"] = poses;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", blended));
		const auto actual = Probe::skin(body.rid(), 0);
		REQUIRE(actual.valid);
		for (int slot = 0; slot < 4; ++slot) {
			CHECK(actual.weights[slot] == weights[slot]);
		}
		const float target = 128 * weights[0] + 256 * weights[1];
		CHECK(Math::abs(target - target / (weights[0] + weights[1])) > 1e-3f);
		REQUIRE(Probe::invoke_pre_step(body.rid()));
		CHECK(Probe::world_position(body.rid(), 0).distance_to(Vector3(target, 0, 0)) <= 1e-4f);
	}

	TEST_CASE("remaining K05 weight termination") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary good = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		Dictionary bad = good.duplicate(true);
		PackedFloat32Array weights = bad["joint_weights"];
		weights.set(0, 0.5f);
		weights.set(2, 0.5f);
		bad["joint_weights"] = weights;
		RemainingSnapshot before(server.ptr(), body.rid());
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
		CHECK(errors.has("SBREM-VALUE"));
		CHECK(errors.has("joint_weights[2]"));
		before.unchanged();
		CHECK(Probe::skinned_count(body.rid()) == 4);
		for (int i = 0; i < 4; ++i) {
			CHECK(Probe::skin(body.rid(), i).weights[0] == 1);
		}
	}

	TEST_CASE("remaining K06 skin indices") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const Dictionary good = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		for (int kind = 0; kind < 5; ++kind) {
			CAPTURE(kind);
			Dictionary bad = good.duplicate(true);
			if (kind < 3) {
				PackedInt32Array selected = bad["vertices"];
				selected.set(3, kind == 0 ? -1 : kind == 1 ? 4
														   : 2);
				bad["vertices"] = selected;
			} else {
				PackedInt32Array joints = bad["joint_indices"];
				joints.set(kind == 3 ? 0 : 3, 1);
				bad["joint_indices"] = joints;
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
			CHECK(errors.has("SBREM-INDEX"));
			CHECK(errors.has("["));
			before.unchanged();
		}
		for (int count : { 1, 1024, 1025 }) {
			CAPTURE(count);
			Dictionary config = good.duplicate(true);
			Array matrices;
			for (int joint = 0; joint < count; ++joint) {
				matrices.push_back(Transform3D());
			}
			config["inv_bind"] = matrices;
			config["initial_pose"] = matrices;
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK(server->soft_body_set_extra_property(body.rid(), "skin/config", config) == (count <= 1024));
			if (count > 1024) {
				CHECK(errors.has("SBREM-SIZE"));
				before.unchanged();
			} else {
				CHECK(Probe::inv_bind_count(body.rid()) == count);
			}
		}
	}

	TEST_CASE("remaining K07 affine matrices") {
		using namespace SoftBodyCapValidation;
		for (double det : { -1e-12, 1e-12, 0.0 }) {
			CHECK_FALSE(determinant(det));
		}
		CHECK(determinant(std::nextafter(1e-12, INFINITY)));
		CHECK(determinant(std::nextafter(-1e-12, -INFINITY)));
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const Dictionary good = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		for (const char *field : { "inv_bind", "initial_pose", "pose" }) {
			for (int kind = 0; kind < 6; ++kind) {
				CAPTURE(field);
				CAPTURE(kind);
				Array matrices;
				Transform3D transform;
				if (kind == 0) {
					matrices.push_back(Variant());
				} else {
					if (kind == 1) {
						transform.origin.x = NAN;
					}
					if (kind == 2) {
						transform.basis = Basis().scaled(Vector3(1, 1, 0));
					}
					if (kind == 3) {
						transform.basis = Basis().scaled(Vector3(1, 1, std::nextafter(1e-12f, 0.0f)));
					}
					if (kind == 4) {
						transform.basis = Basis().scaled(Vector3(1, 1, std::nextafter(1e-12f, INFINITY)));
					}
					if (kind == 5) {
						transform.basis = Basis().scaled(Vector3(-2, 3, 0.5f));
					}
					matrices.push_back(transform);
				}
				Dictionary config = good.duplicate(true);
				config[field] = matrices;
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				const bool pose = String(field) == "pose";
				CHECK(server->soft_body_set_extra_property(body.rid(), pose ? "skin/pose" : "skin/config", pose ? Variant(matrices) : Variant(config)) == (kind >= 4));
				if (kind < 4) {
					CHECK(errors.has(kind == 0 ? "SBREM-TYPE" : "SBREM-VALUE"));
					CHECK(errors.has("[0]"));
					before.unchanged();
				} else {
					REQUIRE(Probe::invoke_pre_step(body.rid()));
					const Dictionary active = server->soft_body_get_extra_property(body.rid(), "skin/config");
					const Array binds = active["inv_bind"], latest = server->soft_body_get_extra_property(body.rid(), "skin/pose");
					const Vector3 points[] = { Vector3(), Vector3(1, 0, 0), Vector3(1, 0, 1), Vector3(0, 0, 1) };
					for (int vertex = 0; vertex < 4; ++vertex) {
						const Vector3 expected = Transform3D(latest[0]).xform(Transform3D(binds[0]).xform(points[vertex]));
						CHECK(Probe::frame(body.rid()).xform(Probe::skin_target(body.rid(), vertex).current).distance_to(expected) <= 1e-4f);
					}
				}
			}
		}
		for (bool existing : { false, true }) {
			CAPTURE(existing);
			MeshBodyScope candidate(server.ptr(), space.rid());
			if (existing) {
				REQUIRE(server->soft_body_set_extra_property(candidate.rid(), "skin/config", good));
			}
			for (bool overflow : { true, false }) {
				CAPTURE(overflow);
				Dictionary config = good.duplicate(true);
				Array matrices;
				matrices.push_back(Transform3D(Basis(), Vector3(overflow ? 2e38f : 1e37f, 0, 0)));
				config["inv_bind"] = matrices;
				config["initial_pose"] = matrices;
				RemainingSnapshot before(server.ptr(), candidate.rid());
				ErrorCapture errors;
				CHECK(server->soft_body_set_extra_property(candidate.rid(), "skin/config", config) == !overflow);
				if (overflow) {
					CHECK(errors.has("SBREM-VALUE"));
					CHECK(errors.has("initial"));
					before.unchanged();
				} else {
					REQUIRE(Probe::invoke_pre_step(candidate.rid()));
					CHECK(Probe::skin_target(candidate.rid(), 0).current.x == doctest::Approx(2e37f).epsilon(1e-5));
				}
			}
		}
	}

	TEST_CASE("remaining K08 coordinate spaces") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
		MeshBodyScope body(server.ptr(), space.rid());
		const Transform3D binding(Basis(Vector3(0, 1, 0), 0.6f), Vector3(3, 4, 5));
		server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, binding);
		Dictionary config = fixture_skin();
		const Transform3D author_inverse(Basis(Vector3(1, 0, 0), 0.4f), Vector3(-2, 1, 0.5f));
		Array inverses;
		inverses.push_back(author_inverse);
		config["inv_bind"] = inverses;
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		Vector<Vector3> bind_points;
		for (int i = 0; i < 4; ++i) {
			bind_points.push_back(binding.xform(Probe::vertex_position(body.rid(), i)));
		}
		class Readback : public PhysicsServer3DRenderingServerHandler {
		public:
			Vector3 vertices[4];
			Vector3 normals[4];
			void set_vertex(int p_index, const Vector3 &p_point) override { vertices[p_index] = p_point; }
			void set_normal(int p_index, const Vector3 &p_normal) override { normals[p_index] = p_normal; }
			void set_aabb(const AABB &) override {}
		};
		Readback *readback = memnew(Readback);
		for (int frame = 0; frame < 60; ++frame) {
			Array pose;
			const Transform3D joint(Basis(Vector3(0, 0, 1), frame < 30 ? 0.0f : 0.3f), frame < 30 ? Vector3() : Vector3(1, 2, 0));
			pose.push_back(joint);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			space.step();
			REQUIRE(Probe::all_finite(body.rid()));
			const Variant public_frame = server->soft_body_get_extra_property(body.rid(), "skin/frame");
			REQUIRE(public_frame.get_type() == Variant::TRANSFORM3D);
			CHECK(Transform3D(public_frame).is_equal_approx(Probe::frame(body.rid())));
			CHECK(Probe::frame(body.rid()).origin.length() > 1.0f);
			server->soft_body_update_rendering_server(body.rid(), readback);
			for (int i = 0; i < 4; ++i) {
				CHECK(Probe::world_position(body.rid(), i).distance_to(joint.xform(author_inverse.xform(bind_points[i]))) <= 1e-4f);
				CHECK(server->soft_body_get_point_global_position(body.rid(), i).distance_to(joint.xform(author_inverse.xform(bind_points[i]))) <= 1e-4f);
				CHECK(readback->vertices[i].distance_to(joint.xform(author_inverse.xform(bind_points[i]))) <= 1e-4f);
				CHECK(readback->normals[i].is_finite());
			}
		}
		memdelete(readback);
	}

	TEST_CASE("remaining K09 pose no rebuild") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config = fixture_skin();
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		space.step();
		const auto id = Probe::body_identity(body.rid()), settings = Probe::settings_identity(body.rid()), generation = Probe::build_counts(body.rid()).generation;
		for (int frame = 1; frame <= 180; ++frame) {
			Array pose;
			pose.push_back(Transform3D(Basis(), Vector3(0, frame * 0.01f, 0)));
			REQUIRE(Probe::set_vertex_local(body.rid(), 0, Probe::vertex_position(body.rid(), 0), Vector3(0.2f, 0.3f, 0.4f)));
			const Vector3 velocity = Probe::velocity(body.rid(), 0);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			CHECK(Probe::velocity(body.rid(), 0) == velocity);
			CHECK(Probe::body_identity(body.rid()) == id);
			CHECK(Probe::settings_identity(body.rid()) == settings);
			CHECK(Probe::build_counts(body.rid()).generation == generation);
			pose[0] = Transform3D(); // Submitted array must own a snapshot.
			space.step();
			CHECK(Probe::all_finite(body.rid()));
			CHECK(Probe::build_counts(body.rid()).skin_init == 1);
			CHECK(Probe::world_position(body.rid(), 0).distance_to(Vector3(0, frame * 0.01f, 0)) <= 1e-4f);
		}
		CHECK(Probe::build_counts(body.rid()).skin_recurring == 180);
	}

	TEST_CASE("remaining K10 pose rejection") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
		MeshBodyScope body(server.ptr(), space.rid());
		Array valid;
		valid.push_back(Transform3D());
		{
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/pose", valid));
			CHECK(errors.has("SBREM-POSE"));
			before.unchanged();
		}
		Dictionary config = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		for (int kind = 0; kind < 4; ++kind) {
			CAPTURE(kind);
			Array pose = valid.duplicate(true);
			if (kind == 0) {
				pose.clear();
			}
			if (kind == 1) {
				pose[0] = true;
			}
			if (kind == 2) {
				pose[0] = Transform3D(Basis().scaled(Vector3(0, 1, 1)), Vector3());
			}
			if (kind == 3) {
				pose[0] = Transform3D(Basis(), Vector3(NAN, 0, 0));
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			CHECK(errors.has(kind == 0 ? "SBREM-POSE" : kind == 1 ? "SBREM-TYPE"
																  : "SBREM-VALUE"));
			before.unchanged();
		}
		{
			MeshBodyScope write_body(server.ptr(), space.rid());
			Dictionary binding = fixture_skin(1, 1);
			binding["max_distance"] = 0.0;
			Array inverses, initial;
			inverses.push_back(Transform3D(Basis(), Vector3(2.0e38f, 0, 0)));
			initial.push_back(Transform3D(Basis(), Vector3(-2.0e38f, 0, 0)));
			binding["inv_bind"] = inverses;
			binding["initial_pose"] = initial;
			REQUIRE(server->soft_body_set_extra_property(write_body.rid(), "skin/config", binding));
			REQUIRE(Probe::set_vertex_local(write_body.rid(), 0, Probe::vertex_position(write_body.rid(), 0), Vector3(1, 2, 3)));
			Array overflow;
			overflow.push_back(Transform3D(Basis(), Vector3(2.0e38f, 0, 0)));
			RemainingSnapshot before(server.ptr(), write_body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(write_body.rid(), "skin/pose", overflow));
			CHECK(errors.has("SBREM-POSE"));
			CHECK(errors.has("write"));
			CHECK(errors.has("["));
			before.unchanged();
			Array near;
			near.push_back(Transform3D(Basis(), Vector3(1.0e38f, 0, 0)));
			errors.clear();
			REQUIRE(server->soft_body_set_extra_property(write_body.rid(), "skin/pose", near));
			CHECK(errors.count() == 0);
			REQUIRE(Probe::invoke_pre_step(write_body.rid()));
			CHECK(Probe::world_position(write_body.rid(), 0).x == doctest::Approx(double(2.0e38f) + double(1.0e38f)).epsilon(1e-5));
		}
		for (bool initialized : { false, true }) {
			CAPTURE(initialized);
			MeshBodyScope fault(server.ptr(), space.rid());
			Dictionary fault_skin = fixture_skin(1, 1);
			fault_skin["max_distance"] = 0.0;
			REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/config", fault_skin));
			if (initialized) {
				space.step();
			}
			Array far_pose;
			far_pose.push_back(Transform3D(Basis(Vector3(2.5e38f, 2.5e38f, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3()));
			REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/pose", far_pose));
			REQUIRE(Probe::set_body_frame(fault.rid(), Transform3D(Basis(Vector3(0, 0, 1), Math::PI / 4), Vector3(3, 4, 5))));
			const auto generation = Probe::build_counts(fault.rid()).generation;
			const auto init_count = Probe::build_counts(fault.rid()).skin_init;
			ErrorCapture errors;
			space.step();
			CHECK(errors.has("SBREM-POSE"));
			CHECK(errors.has("pre_step"));
			CHECK(errors.has("["));
			CHECK_FALSE(Probe::in_space(fault.rid()));
			CHECK(Probe::build_counts(fault.rid()).generation == generation);
			CHECK(Probe::build_counts(fault.rid()).skin_init == init_count);
			CHECK(Array(server->soft_body_get_extra_property(fault.rid(), "skin/pose")) == far_pose);
			errors.clear();
			space.step(3);
			CHECK(errors.count() == 0);
			REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/pose", valid));
			CHECK_FALSE(Probe::in_space(fault.rid()));
			REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/config", fault_skin));
			CHECK_FALSE(Probe::in_space(fault.rid()));
			server->soft_body_set_space(fault.rid(), RID());
			server->soft_body_set_space(fault.rid(), space.rid());
			space.step();
			CHECK(Probe::in_space(fault.rid()));
			CHECK(Probe::all_finite(fault.rid()));
			CHECK(Probe::build_counts(fault.rid()).generation == generation + 1);
			CHECK(Probe::build_counts(fault.rid()).skin_init == init_count + 1);
		}
		// Nearby finite case uses the same nonzero COM/rotation but a smaller
		// matrix. Only source vertex 0 is skinned, so world targets remain bounded
		// while the scratch matrix conversion straddles the overflow boundary.
		{
			MeshBodyScope near(server.ptr(), space.rid());
			Dictionary near_skin = fixture_skin(1, 1);
			near_skin["max_distance"] = 0.0;
			REQUIRE(server->soft_body_set_extra_property(near.rid(), "skin/config", near_skin));
			Array pose;
			pose.push_back(Transform3D(Basis(Vector3(2e38f, 2e38f, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3()));
			REQUIRE(server->soft_body_set_extra_property(near.rid(), "skin/pose", pose));
			REQUIRE(Probe::set_body_frame(near.rid(), Transform3D(Basis(Vector3(0, 0, 1), Math::PI / 4), Vector3(3, 4, 5))));
			ErrorCapture errors;
			space.step();
			CHECK(errors.count() == 0);
			CHECK(Probe::in_space(near.rid()));
			CHECK(Probe::all_finite(near.rid()));
			CHECK(Probe::build_counts(near.rid()).skin_init == 1);
			CHECK(Probe::build_counts(near.rid()).solver_steps == 1);
		}
	}

	TEST_CASE("remaining K11 hold last pose") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		Vector<Vector3> held;
		for (bool resend : { false, true }) {
			CAPTURE(resend);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config = fixture_skin();
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
			Array pose;
			pose.push_back(Transform3D(Basis(), Vector3(0, 1, 0)));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			ErrorCapture errors;
			for (int frame = 0; frame < 180; ++frame) {
				if (resend) {
					REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
				}
				space.step();
				REQUIRE(Probe::all_finite(body.rid()));
				const auto target = Probe::skin_target(body.rid(), 0);
				CHECK(Probe::frame(body.rid()).xform(target.current).distance_to(Vector3(0, 1, 0)) <= 1e-4f);
				if (!resend) {
					held.push_back(Probe::world_position(body.rid(), 0));
				} else {
					CHECK(Probe::world_position(body.rid(), 0).distance_to(held[frame]) <= 1e-4f);
				}
			}
			CHECK(errors.count() == 0);
			CHECK(Probe::build_counts(body.rid()).skin_init == 1);
			CHECK(Probe::build_counts(body.rid()).skin_recurring == 179);
		}
	}

	TEST_CASE("remaining K12 absent skin") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary scalar;
		scalar["friction"] = 0.2;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", scalar));
		space.step(30);
		CHECK(Probe::build_counts(body.rid()).pre_step == 30);
		CHECK(Probe::build_counts(body.rid()).skin_init == 0);
		CHECK(Probe::build_counts(body.rid()).skin_recurring == 0);
		CHECK(Probe::skinned_count(body.rid()) == 0);
		CHECK(Probe::edge_count(body.rid()) > 0);
	}

	TEST_CASE("remaining K13 hard skin lifetime") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr()), second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		server->soft_body_pin_point(body.rid(), 3, true);
		Dictionary config = fixture_skin(1, 2);
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		for (int change = 0; change < 3; ++change) {
			CAPTURE(change);
			if (change == 1) {
				server->soft_body_set_total_mass(body.rid(), 2.0);
			}
			if (change == 2) {
				server->soft_body_set_space(body.rid(), second.rid());
			}
			REQUIRE(Probe::in_space(body.rid()));
			for (int i : { 0, 1, 3 }) {
				CHECK(Probe::shared_inv_mass(body.rid(), i) == 0);
				CHECK(Probe::vertex_inv_mass(body.rid(), i) == 0);
			}
			if (change < 2) {
				first.step(30);
			} else {
				second.step(30);
			}
			CHECK(Probe::world_position(body.rid(), 0).length() <= 1e-4f);
			CHECK(Probe::world_position(body.rid(), 1).distance_to(Vector3(1, 0, 0)) <= 1e-4f);
		}
	}

	TEST_CASE("remaining K14 pin conflict") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool alias : { false, true }) {
			for (bool pin_first : { false, true }) {
				CAPTURE(alias);
				CAPTURE(pin_first);
				const auto fixture = fixture_seam();
				MeshBodyScope body(server.ptr(), space.rid(), alias ? &fixture : nullptr);
				Dictionary scalar;
				scalar["friction"] = 1.0;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", scalar));
				Dictionary skin = fixture_skin(1, alias ? 6 : 2);
				skin["max_distance"] = 0.0;
				const int pin = alias ? 4 : 0;
				if (pin_first) {
					server->soft_body_pin_point(body.rid(), pin, true);
				} else {
					REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
				}
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				if (pin_first) {
					CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
				} else {
					server->soft_body_pin_point(body.rid(), pin, true);
				}
				CHECK(errors.has("SBREM-CONFLICT"));
				CHECK(server->soft_body_is_point_pinned(body.rid(), pin) == pin_first);
				before.unchanged();
				errors.clear();
				skin["max_distance"] = 0.05;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
				server->soft_body_pin_point(body.rid(), pin, true);
				CHECK(errors.count() == 0);
				CHECK(server->soft_body_is_point_pinned(body.rid(), pin));
			}
		}
	}

	TEST_CASE("remaining K15 skin distance") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (double distance : { 0.0, 0.05, double(FLT_MAX) }) {
			CAPTURE(distance);
			SpaceScope space(server.ptr());
			server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 9.8);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config = fixture_skin();
			config["max_distance"] = distance;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
			space.step();
			server->soft_body_apply_central_impulse(body.rid(), Vector3(0, -1, 0));
			space.step(180);
			REQUIRE(Probe::all_finite(body.rid()));
			for (int i = 0; i < 4; ++i) {
				const auto target = Probe::skin_target(body.rid(), i);
				const double displacement = Probe::vertex_position(body.rid(), i).distance_to(target.current);
				if (distance == FLT_MAX) {
					CHECK(displacement > 0.1);
				} else {
					CHECK(displacement <= distance + 1e-4);
				}
			}
		}
	}

	TEST_CASE("remaining K16 partial normals") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config = fixture_skin(1, 1);
		config["back_stop_distance"] = 0.0;
		{
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
			CHECK(errors.has("SBREM-SKIN-NORMAL"));
			CHECK(errors.has("vertices[0]"));
			before.unchanged();
		}
		config["back_stop_distance"] = double(FLT_MAX);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		CHECK(Probe::skinned_count(body.rid()) == 1);
		CHECK(Probe::skin(body.rid(), 0).normal_info == 0);
	}

	TEST_CASE("remaining K17 normal derivation") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config = fixture_skin();
		config["back_stop_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		REQUIRE(Probe::invoke_pre_step(body.rid()));
		for (int i = 0; i < 4; ++i) {
			const auto skin = Probe::skin(body.rid(), i);
			CHECK((skin.normal_info >> 24) == ((skin.vertex == 0 || skin.vertex == 2) ? 2 : 1));
			const auto target = Probe::skin_target(body.rid(), skin.vertex);
			CHECK(target.normal.is_finite());
			CHECK(target.normal.distance_to(Vector3(0, 1, 0)) <= 1e-5f);
		}
	}

	TEST_CASE("remaining K18 normal bounds") {
		using namespace SoftBodyCapValidation;
		for (uint64_t start : { (uint64_t(1) << 24) - 1, uint64_t(1) << 24 }) {
			for (uint64_t count : { uint64_t(255), uint64_t(256) }) {
				CAPTURE(start);
				CAPTURE(count);
				uint32_t packed = 0;
				String error;
				const bool valid = start < (uint64_t(1) << 24) && count < 256;
				CHECK(normal_info(start, count, packed, &error) == valid);
				if (valid) {
					CHECK((packed & 0xffffff) == start);
					CHECK((packed >> 24) == count);
				} else {
					CHECK(error.contains("SBREM-SKIN-NORMAL"));
					CHECK(error.contains("count"));
					CHECK(error.contains("start"));
				}
			}
		}
	}

	TEST_CASE("remaining K19 skin aliases") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_seam();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		const Dictionary good = fixture_skin(2, 6);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", good));
		CHECK(Probe::skinned_count(body.rid()) == 4);
		HashSet<int> vertices;
		for (int i = 0; i < 4; ++i) {
			CHECK_FALSE(vertices.has(Probe::skin(body.rid(), i).vertex));
			vertices.insert(Probe::skin(body.rid(), i).vertex);
		}
		for (int kind = 0; kind < 4; ++kind) {
			CAPTURE(kind);
			Dictionary bad = good.duplicate(true);
			if (kind == 0) {
				PackedInt32Array joints = bad["joint_indices"];
				joints.set(16, 1);
				bad["joint_indices"] = joints;
			} else if (kind == 1) {
				bad = fixture_skin(2, 5);
			}
			if (kind == 2) {
				PackedFloat32Array weights = bad["joint_weights"];
				weights.set(16, 0.25f);
				weights.set(17, 0.75f);
				bad["joint_weights"] = weights;
			}
			if (kind == 3) {
				PackedInt32Array joints = bad["joint_indices"];
				joints.set(18, 1); // Valid joint, zero-weight slot still belongs to tuple.
				bad["joint_indices"] = joints;
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", bad));
			CHECK(errors.has("SBREM-ALIAS"));
			CHECK(errors.has("source vertices"));
			before.unchanged();
		}
	}

	TEST_CASE("remaining K20 skin state machine") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr()), second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		Dictionary config = fixture_skin();
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		Array a, b, c;
		a.push_back(Transform3D(Basis(), Vector3(0, 1, 0)));
		b.push_back(Transform3D(Basis(), Vector3(0, 2, 0)));
		c.push_back(Transform3D(Basis(), Vector3(0, 3, 0)));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", a));
		config["initial_pose"] = b;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == b);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", c));
		first.step();
		uint64_t init = Probe::build_counts(body.rid()).skin_init;
		server->soft_body_set_space(body.rid(), second.rid());
		CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == c);
		second.step();
		CHECK(Probe::build_counts(body.rid()).skin_init == ++init);
		auto fixture = fixture_hinge();
		fixture.vertices.set(1, Vector3(2, 0, 0));
		RID new_mesh = fixture.create_mesh();
		server->soft_body_set_mesh(body.rid(), new_mesh);
		CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == c);
		second.step();
		CHECK(Probe::build_counts(body.rid()).skin_init == ++init);
		for (int i = 0; i < 4; ++i) {
			CHECK(server->soft_body_get_point_global_position(body.rid(), i).distance_to(fixture.vertices[i] + Vector3(0, 3, 0)) <= 1e-4f);
		}
		RemainingMeshFixture incompatible;
		for (const Vector3 &point : { Vector3(), Vector3(1, 0, 0), Vector3(0, 0, 1) }) {
			incompatible.vertices.push_back(point);
		}
		for (int index : { 0, 1, 2 }) {
			incompatible.indices.push_back(index);
		}
		RID bad_mesh = incompatible.create_mesh();
		const auto generation = Probe::build_counts(body.rid()).generation;
		{
			ErrorCapture errors;
			server->soft_body_set_mesh(body.rid(), bad_mesh);
			CHECK(errors.has("SBREM-INDEX"));
			CHECK_FALSE(Probe::in_space(body.rid()));
			CHECK(Probe::build_counts(body.rid()).generation == generation);
			CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == c);
		}
		server->soft_body_set_mesh(body.rid(), new_mesh);
		second.step();
		CHECK(Probe::build_counts(body.rid()).generation == generation + 1);
		CHECK(Probe::build_counts(body.rid()).skin_init == ++init);
		CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == c);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", Variant()));
		CHECK(server->soft_body_get_extra_property(body.rid(), "skin/pose").get_type() == Variant::NIL);
		CHECK(Probe::skinned_count(body.rid()) == 0);
		server->soft_body_set_mesh(body.rid(), body.mesh_rid());
		RenderingServer::get_singleton()->free_rid(new_mesh);
		RenderingServer::get_singleton()->free_rid(bad_mesh);
	}

	TEST_CASE("remaining K21 transform policy") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const Transform3D binding(Basis(Vector3(0, 1, 0), 0.4f), Vector3(1, 2, 3));
		server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, binding);
		Dictionary config = fixture_skin();
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		REQUIRE(Probe::invoke_pre_step(body.rid()));
		CHECK(Probe::world_position(body.rid(), 0).distance_to(binding.origin) <= 1e-4f);
		const Transform3D frame = Probe::frame(body.rid());
		RemainingSnapshot before(server.ptr(), body.rid());
		ErrorCapture errors;
		server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D(Basis(), Vector3(1, 0, 0)));
		CHECK(errors.has("SBREM-CONTEXT"));
		CHECK(Probe::frame(body.rid()) == frame);
		before.unchanged();
	}

	TEST_CASE("remaining K22 pose wake") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config = fixture_skin();
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		space.step();
		const auto id = Probe::body_identity(body.rid());
		server->soft_body_set_state(body.rid(), PS3DE::BODY_STATE_SLEEPING, true);
		CHECK(bool(server->soft_body_get_state(body.rid(), PS3DE::BODY_STATE_SLEEPING)));
		Array pose;
		pose.push_back(Transform3D(Basis(), Vector3(0, 1, 0)));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
		CHECK_FALSE(bool(server->soft_body_get_state(body.rid(), PS3DE::BODY_STATE_SLEEPING)));
		space.step();
		CHECK(Probe::body_identity(body.rid()) == id);
		CHECK(Probe::world_position(body.rid(), 0).distance_to(Vector3(0, 1, 0)) <= 1e-4f);
	}

	TEST_CASE("remaining K23 backstop behavior") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (const char *field : { "max_distance", "back_stop_distance", "back_stop_radius" }) {
			SpaceScope space(server.ptr());
			MeshBodyScope body(server.ptr(), space.rid());
			for (double value : { 0.0, -1.0, double(NAN), double(INFINITY), double(FLT_MAX), double(FLT_MAX) * 2 }) {
				CAPTURE(field);
				CAPTURE(value);
				Dictionary config = fixture_skin();
				config[field] = value;
				const bool valid = std::isfinite(value) && value <= FLT_MAX && (String(field) == "back_stop_radius" ? value > 0 : value >= 0);
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK(server->soft_body_set_extra_property(body.rid(), "skin/config", config) == valid);
				if (!valid) {
					CHECK(errors.has("SBREM-VALUE"));
					CHECK(errors.has(field));
					before.unchanged();
				}
			}
		}
		for (bool enabled : { false, true }) {
			for (bool inside : { false, true }) {
				CAPTURE(enabled);
				CAPTURE(inside);
				SpaceScope space(server.ptr());
				server->area_set_param(space.rid(), PS3DE::AREA_PARAM_GRAVITY, 0.0);
				MeshBodyScope body(server.ptr(), space.rid());
				Dictionary config = fixture_skin();
				config["max_distance"] = double(FLT_MAX);
				config["back_stop_distance"] = enabled ? 0.0 : double(FLT_MAX);
				config["back_stop_radius"] = 0.5;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
				space.step();
				for (int i = 0; i < 4; ++i) {
					REQUIRE(Probe::set_vertex_local(body.rid(), i, Probe::vertex_position(body.rid(), i) + Vector3(0, inside ? -0.25f : 0.25f, 0)));
				}
				space.step();
				REQUIRE(Probe::all_finite(body.rid()));
				const float y = Probe::world_position(body.rid(), 0).y;
				if (inside && enabled) {
					CHECK(y >= -1e-4f);
				} else {
					CHECK(y == doctest::Approx(inside ? -0.25f : 0.25f).epsilon(1e-4));
				}
			}
		}
	}

	TEST_CASE("remaining L08 skin anchors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr()), second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		Dictionary skin = fixture_skin(1, 1);
		skin["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
		Dictionary lra;
		lra["type"] = 1;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", lra));
		for (int change = 0; change < 3; ++change) {
			CAPTURE(change);
			if (change == 1) {
				server->soft_body_set_total_mass(body.rid(), 2);
			}
			if (change == 2) {
				server->soft_body_set_space(body.rid(), second.rid());
			}
			CHECK(Probe::lra_count(body.rid()) == 3);
			CHECK(Probe::vertex_inv_mass(body.rid(), 0) == 0);
			for (int i = 0; i < 3; ++i) {
				const auto constraint = Probe::lra(body.rid(), i);
				CHECK(constraint.vertices[0] == 0);
				CHECK(constraint.rest == doctest::Approx(Probe::vertex_position(body.rid(), constraint.vertices[1]).length()).epsilon(1e-5));
			}
		}
	}

	TEST_CASE("remaining A07 unused vertices") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_seam();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		Dictionary vertex;
		PackedFloat32Array scales = make_floats(7, 1);
		scales.set(6, 2);
		vertex["edge_scale"] = scales;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", vertex));
		CHECK(Probe::vertex_count(body.rid()) == 4);
		{
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			scales.set(6, NAN);
			vertex["edge_scale"] = scales;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "vertex/config", vertex));
			CHECK(errors.has("SBREM-VALUE"));
			CHECK(errors.has("[6]"));
			before.unchanged();
		}
		{
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", fixture_skin(1, 7)));
			CHECK(errors.has("SBREM-INDEX"));
			CHECK(errors.has("6"));
			before.unchanged();
		}
		{
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			server->soft_body_pin_point(body.rid(), 6, true);
			CHECK(errors.has("SBREM-INDEX"));
			CHECK_FALSE(server->soft_body_is_point_pinned(body.rid(), 6));
			before.unchanged();
		}
	}

	TEST_CASE("remaining P02 wrong key") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const char *flat[] = { "scalar/friction", "bend/compliance", "lra/type", "vertex/edge_scale", "volume/tetrahedra", "skin/joint_weights", "rod/joints" };
		for (int cap = 0; cap < 7; ++cap) {
			for (bool prototype : { false, true }) {
				const String key = prototype ? String(flat[cap]) : String(REMAINING_CONFIG_KEYS[cap]).get_slice("/", 0) + "/confg"; // codespell:ignore confg
				CAPTURE(key);
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), key, 1));
				CHECK(errors.has("SBREM-KEY"));
				CHECK(errors.has(key));
				CHECK(server->soft_body_get_extra_property(body.rid(), key).get_type() == Variant::NIL);
				before.unchanged();
			}
		}
	}

	TEST_CASE("remaining P03 top type") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (const char *key : REMAINING_CONFIG_KEYS) {
			for (const Variant &value : { Variant(1), Variant(Array()), Variant("bad"), Variant(true) }) {
				CAPTURE(key);
				CAPTURE(value.get_type());
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), key, value));
				CHECK(errors.has("SBREM-TYPE"));
				CHECK(errors.has(key));
				before.unchanged();
			}
		}
		RemainingSnapshot before(server.ptr(), body.rid());
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/pose", Dictionary()));
		CHECK(errors.has("SBREM-TYPE"));
		CHECK(errors.has("skin/pose"));
		before.unchanged();
	}

	TEST_CASE("remaining P04 strict schema") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (int cap = 0; cap < 7; ++cap) {
			CAPTURE(REMAINING_CONFIG_KEYS[cap]);
			const Dictionary good = remaining_config(cap);
			Array candidates;
			Dictionary typo = good.duplicate(true);
			typo["unknown_field"] = 1.0;
			candidates.push_back(typo);
			candidates.push_back(Dictionary());
			if (cap == 1 || cap == 2 || cap == 4 || cap == 5 || cap == 6) {
				for (const Variant *field = good.next(nullptr); field != nullptr; field = good.next(field)) {
					if (cap == 6 && String(*field) != "joints") {
						continue;
					}
					Dictionary missing = good.duplicate(true);
					missing.erase(*field);
					candidates.push_back(missing);
				}
			}
			for (int i = 0; i < candidates.size(); ++i) {
				CAPTURE(i);
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], candidates[i]));
				CHECK(errors.has("SBREM-KEY"));
				CHECK(errors.has(REMAINING_CONFIG_KEYS[cap]));
				before.unchanged();
			}
		}
	}

	TEST_CASE("remaining P05 inner types") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (int cap = 0; cap < 7; ++cap) {
			Dictionary full = remaining_config(cap);
			if (cap == 0) {
				full["gravity_factor"] = 1.0;
				full["vertex_radius"] = 0.0;
				full["faces_double_sided"] = false;
			}
			if (cap == 1) {
				full["type"] = 1;
			}
			if (cap == 2) {
				full["max_distance_multiplier"] = 1.0;
			}
			if (cap == 4) {
				full["compliance"] = 0.0;
				full["fixed"] = PackedInt32Array();
			}
			if (cap == 5) {
				full["max_distance"] = 0.05;
				full["back_stop_distance"] = double(FLT_MAX);
				full["back_stop_radius"] = 40.0;
			}
			for (const Variant *field = full.next(nullptr); field != nullptr; field = full.next(field)) {
				CAPTURE(REMAINING_CONFIG_KEYS[cap]);
				CAPTURE(String(*field));
				const Variant original = full[*field];
				Dictionary bad = full.duplicate(true);
				bad[*field] = original.get_type() == Variant::BOOL ? Variant(0) : original.get_type() == Variant::FLOAT ? Variant(1)
						: original.get_type() == Variant::INT															? Variant(1.0)
																														: Variant(true);
				RemainingSnapshot before(server.ptr(), body.rid());
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], bad));
				CHECK(errors.has("SBREM-TYPE"));
				CHECK(errors.has(String(*field)));
				before.unchanged();
				if (original.get_type() == Variant::ARRAY) {
					errors.clear();
					Array matrices;
					matrices.push_back(Variant());
					bad[*field] = matrices;
					CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], bad));
					CHECK(errors.has("SBREM-TYPE"));
					CHECK(errors.has(String(*field) + "[0]"));
					before.unchanged();
				}
			}
		}
	}

	TEST_CASE("remaining P06 finite ranges") {
		struct Numeric {
			int cap;
			const char *field;
			bool positive;
			double maximum;
			bool array;
		};
		const Numeric fields[] = { { 0, "friction", false, FLT_MAX, false }, { 0, "restitution", false, 1, false }, { 0, "gravity_factor", false, FLT_MAX, false }, { 0, "vertex_radius", false, FLT_MAX, false }, { 1, "compliance", false, FLT_MAX, true }, { 2, "max_distance_multiplier", true, FLT_MAX, true }, { 3, "edge_scale", true, FLT_MAX, true }, { 3, "shear_scale", true, FLT_MAX, true }, { 4, "compliance", false, FLT_MAX, false }, { 5, "max_distance", false, FLT_MAX, false }, { 5, "back_stop_distance", false, FLT_MAX, false }, { 5, "back_stop_radius", true, FLT_MAX, false } };
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (const auto &field : fields) {
			const RID body = server->soft_body_create();
			const Dictionary base = remaining_config(field.cap);
			for (int shape = 0; shape < (field.array ? 2 : 1); ++shape) {
				for (int bad_index = 0; bad_index < (shape ? 3 : 1); ++bad_index) {
					CAPTURE(bad_index);
					for (double value : { 0.0, -1.0, double(std::numeric_limits<float>::denorm_min()), field.maximum, field.maximum * 2, double(NAN), double(INFINITY), -double(INFINITY), double(FLT_MAX) * 2 }) {
						CAPTURE(field.field);
						CAPTURE(shape);
						CAPTURE(value);
						Dictionary config = base.duplicate(true);
						// Packed arrays cannot carry finite double-overflow values; their
						// converted INF still exercises the array element finite guard.
						PackedFloat32Array elements = make_floats(3, 1.0f);
						elements.set(bad_index, float(value));
						config[field.field] = shape == 0 ? Variant(value) : Variant(elements);
						const bool valid = std::isfinite(value) && value <= field.maximum && (field.positive ? value > 0 : value >= 0);
						RemainingSnapshot before(server.ptr(), body);
						ErrorCapture errors;
						CHECK(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[field.cap], config) == valid);
						if (!valid) {
							CHECK(errors.has("SBREM-VALUE"));
							CHECK(errors.has(field.field));
							if (shape) {
								CHECK(errors.has(vformat("[%d]", bad_index)));
							}
							before.unchanged();
						}
					}
				}
			}
			server->free_rid(body);
		}
		{
			const RID body = server->soft_body_create();
			REQUIRE(set_valid_rod(server.ptr(), body, 4));
			for (const char *field : { "compliance", "bend" }) {
				for (float value : { -1.0f, NAN, INFINITY, float(double(FLT_MAX) * 2) }) {
					Dictionary config = rod_config(4);
					config[field] = make_floats(String(field) == "bend" ? 2 : 3, value);
					RemainingSnapshot before(server.ptr(), body);
					ErrorCapture errors;
					CHECK_FALSE(server->soft_body_set_extra_property(body, "rod/config", config));
					CHECK(errors.has("SBREM-VALUE"));
					CHECK(errors.has(field));
					before.unchanged();
				}
			}
			for (int64_t fixed : { INT64_MIN, int64_t(-1), int64_t(5), int64_t(1) << 32, INT64_MAX }) {
				Dictionary config = rod_config(4);
				config["fixed"] = fixed;
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				CHECK_FALSE(server->soft_body_set_extra_property(body, "rod/config", config));
				CHECK(errors.has("SBREM-VALUE"));
				CHECK(errors.has("fixed"));
				before.unchanged();
			}
			server->free_rid(body);
		}
		for (int cap : { 1, 2 }) {
			for (int value : { -1, 0, 1, 2, 3, INT32_MAX }) {
				CAPTURE(cap);
				CAPTURE(value);
				const RID body = server->soft_body_create();
				Dictionary config = remaining_config(cap);
				config["type"] = value;
				ErrorCapture errors;
				const bool valid = cap == 1 ? value >= 0 && value <= 1 : value >= 1 && value <= 2;
				CHECK(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[cap], config) == valid);
				if (!valid) {
					CHECK(errors.has("SBREM-VALUE"));
					CHECK(errors.has("type"));
				}
				server->free_rid(body);
			}
		}
	}

	TEST_CASE("remaining P07 snapshots isolated") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int cap = 0; cap < 7; ++cap) {
			CAPTURE(cap);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary original = remaining_config(cap);
			if (cap == 1) {
				original["compliance"] = make_floats(2, 1e-4);
			}
			if (cap == 3) {
				original["edge_scale"] = make_floats(2, 2);
			}
			if (cap == 4) {
				PackedInt32Array tetra = original["tetrahedra"];
				tetra.set(1, 2);
				tetra.set(2, 1);
				original["tetrahedra"] = tetra;
			}
			if (cap == 5) {
				PackedFloat32Array weights = original["joint_weights"];
				weights.set(0, 0.999995f);
				original["joint_weights"] = weights;
			}
			const Dictionary authored = original.duplicate(true);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], original));
			CHECK(SoftBodyCapValidation::equal(server->soft_body_get_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap]), authored));
			RemainingSnapshot before(server.ptr(), body.rid());
			if (cap == 5) {
				Array nested = original["initial_pose"];
				nested[0] = Transform3D(Basis(), Vector3(9, 9, 9));
			}
			original.clear();
			before.unchanged();
			Dictionary returned = server->soft_body_get_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap]);
			if (cap == 5) {
				Array nested = returned["inv_bind"];
				nested[0] = Transform3D(Basis(), Vector3(8, 8, 8));
			}
			returned.clear();
			before.unchanged();
			if (cap == 5) {
				Array pose;
				pose.push_back(Transform3D(Basis(), Vector3(0, 1, 0)));
				const Array expected = pose.duplicate(true);
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
				RemainingSnapshot after(server.ptr(), body.rid());
				pose[0] = Transform3D();
				Array read = server->soft_body_get_extra_property(body.rid(), "skin/pose");
				read.clear();
				CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == expected);
				after.unchanged();
			}
		}
	}

	TEST_CASE("remaining P08 reject atomic") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary skin = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
		Array pose;
		pose.push_back(Transform3D(Basis(), Vector3(0, 1, 0)));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
		Dictionary vertex;
		vertex["edge_scale"] = 1.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", vertex));
		space.step(3);
		Dictionary bad = vertex.duplicate(true);
		bad["edge_scale"] = make_floats(5, 2);
		RemainingSnapshot before(server.ptr(), body.rid());
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "vertex/config", bad));
		CHECK(errors.has("SBREM-SIZE"));
		CHECK(errors.has("edge_scale"));
		before.unchanged();
		{
			MeshBodyScope rod(server.ptr(), space.rid());
			server->soft_body_pin_point(rod.rid(), 0, true);
			RemainingSnapshot pinned(server.ptr(), rod.rid());
			errors.clear();
			CHECK_FALSE(set_valid_rod(server.ptr(), rod.rid(), 4));
			CHECK(errors.has("SBREM-CONFLICT"));
			CHECK(server->soft_body_is_point_pinned(rod.rid(), 0));
			pinned.unchanged();
			server->soft_body_remove_all_pinned_points(rod.rid());
			REQUIRE(set_valid_rod(server.ptr(), rod.rid(), 4));
			space.step(3);
			RemainingSnapshot live(server.ptr(), rod.rid());
			const Variant values[] = { make_joints(4), make_floats(3), make_floats(2), 1 };
			const char *keys[] = { "rod/joints", "rod/compliance", "rod/bend", "rod/fixed" };
			for (int i = 0; i < 4; ++i) {
				for (const Variant &value : { values[i], Variant() }) {
					errors.clear();
					CHECK_FALSE(server->soft_body_set_extra_property(rod.rid(), keys[i], value));
					CHECK(errors.has("SBREM-KEY"));
					CHECK(server->soft_body_get_extra_property(rod.rid(), keys[i]).get_type() == Variant::NIL);
					live.unchanged();
				}
			}
			errors.clear();
			server->soft_body_pin_point(rod.rid(), 0, true);
			CHECK(errors.has("SBREM-CONFLICT"));
			CHECK_FALSE(server->soft_body_is_point_pinned(rod.rid(), 0));
			server->soft_body_pin_point(rod.rid(), 99, false);
			server->soft_body_remove_all_pinned_points(rod.rid());
			live.unchanged();
			server->soft_body_set_linear_stiffness(rod.rid(), 0);
			RemainingSnapshot zero_stiffness(server.ptr(), rod.rid());
			errors.clear();
			CHECK_FALSE(server->soft_body_set_extra_property(rod.rid(), "rod/config", Variant()));
			CHECK(errors.has("SBREM-CONTEXT"));
			zero_stiffness.unchanged();
		}
	}

	TEST_CASE("remaining P09 clear semantics") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (int cap = 0; cap < 7; ++cap) {
			CAPTURE(cap);
			MeshBodyScope body(server.ptr(), space.rid());
			server->soft_body_set_space(body.rid(), RID());
			server->soft_body_set_space(body.rid(), space.rid());
			const Dictionary defaults = server->soft_body_get_extra_property(body.rid(), "scalar/current");
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], remaining_config(cap)));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], Variant()));
			CHECK(server->soft_body_get_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap]).get_type() == Variant::NIL);
			CHECK(Probe::dihedral_bend_count(body.rid()) == 0);
			CHECK(Probe::lra_count(body.rid()) == 0);
			CHECK(Probe::volume_count(body.rid()) == 0);
			CHECK(Probe::skinned_count(body.rid()) == 0);
			CHECK(Probe::rod_count(body.rid()) == 0);
			CHECK(Probe::update_position(body.rid()));
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/current")) == defaults);
			CHECK(server->soft_body_get_extra_property(body.rid(), "skin/pose").get_type() == Variant::NIL);
			RemainingSnapshot before(server.ptr(), body.rid());
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], Variant()));
			before.unchanged();
		}
		const RID rod = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), rod, 4));
		server->soft_body_set_space(rod, space.rid());
		REQUIRE(Probe::in_space(rod));
		CHECK_FALSE(Probe::update_position(rod));
		REQUIRE(server->soft_body_set_extra_property(rod, "rod/config", Variant()));
		CHECK_FALSE(Probe::in_space(rod));
		CHECK(server->soft_body_get_extra_property(rod, "rod/config").get_type() == Variant::NIL);
		const RID mesh = make_quad_mesh();
		server->soft_body_set_mesh(rod, mesh);
		REQUIRE(Probe::in_space(rod));
		CHECK(Probe::update_position(rod));
		server->free_rid(rod);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	TEST_CASE("remaining P10 property metadata") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		const auto list = server->soft_body_get_extra_property_list(body.rid());
		for (const char *key : REMAINING_CONFIG_KEYS) {
			const Dictionary entry = find_property(list, key);
			REQUIRE_FALSE(entry.is_empty());
			CHECK(int(entry["type"]) == Variant::DICTIONARY);
			CHECK((int(entry["usage"]) & PROPERTY_USAGE_STORAGE) != 0);
		}
		const Dictionary pose = find_property(list, "skin/pose");
		REQUIRE_FALSE(pose.is_empty());
		CHECK(int(pose["type"]) == Variant::ARRAY);
		CHECK((int(pose["usage"]) & PROPERTY_USAGE_STORAGE) == 0);
		const Variant::Type expected_types[] = { Variant::DICTIONARY, Variant::INT, Variant::INT, Variant::INT, Variant::INT, Variant::FLOAT, Variant::PACKED_VECTOR3_ARRAY, Variant::PACKED_INT32_ARRAY, Variant::TRANSFORM3D, Variant::PACKED_FLOAT32_ARRAY };
		int live_index = 0;
		for (const char *key : REMAINING_LIVE_KEYS) {
			CAPTURE(key);
			const Dictionary entry = find_property(list, key);
			REQUIRE_FALSE(entry.is_empty());
			CHECK((int(entry["usage"]) & PROPERTY_USAGE_STORAGE) == 0);
			const Variant value = server->soft_body_get_extra_property(body.rid(), key);
			CHECK(int(entry["type"]) == expected_types[live_index]);
			CHECK(value.get_type() == expected_types[live_index]);
			++live_index;
			CHECK(value.get_type() == int(entry["type"]));
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), key, value));
			CHECK(errors.has("read-only"));
			before.unchanged();
		}
	}

	TEST_CASE("remaining P11 read lifetime") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (int cap = 0; cap < 7; ++cap) {
			CAPTURE(cap);
			const RID body = server->soft_body_create();
			const Dictionary config = remaining_config(cap);
			REQUIRE(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[cap], config));
			CHECK(Dictionary(server->soft_body_get_extra_property(body, REMAINING_CONFIG_KEYS[cap])) == config);
			if (cap == 5) {
				CHECK(Array(server->soft_body_get_extra_property(body, "skin/pose")) == Array(config["initial_pose"]));
			}
			for (const char *key : REMAINING_LIVE_KEYS) {
				CAPTURE(key);
				ErrorCapture errors;
				CHECK(server->soft_body_get_extra_property(body, key).get_type() == Variant::NIL);
				CHECK(errors.count() == 1);
				CHECK(errors.has("without a physics space"));
			}
			server->free_rid(body);
		}
	}

	TEST_CASE("remaining P12 backend discovery") {
		for (const char *name : { "Jolt Physics", "GodotPhysics3D" }) {
			CAPTURE(name);
			ServerScope server(name);
			REQUIRE(server.is_valid());
			const RID body = server->soft_body_create();
			const auto list = server->soft_body_get_extra_property_list(body);
			HashSet<String> names;
			for (int i = 0; i < list.size(); ++i) {
				const String key = Dictionary(list[i])["name"];
				CHECK_FALSE(names.has(key));
				names.insert(key);
			}
			for (const char *key : REMAINING_CONFIG_KEYS) {
				CHECK(names.has(key) == (String(name) == "Jolt Physics"));
			}
			for (const char *key : REMAINING_LIVE_KEYS) {
				CHECK(names.has(key) == (String(name) == "Jolt Physics"));
			}
			CHECK(names.has("skin/pose") == (String(name) == "Jolt Physics"));
			if (String(name) != "Jolt Physics") {
				ErrorCapture errors;
				for (const char *key : REMAINING_CONFIG_KEYS) {
					CHECK_FALSE(server->soft_body_set_extra_property(body, key, Dictionary()));
					CHECK(server->soft_body_get_extra_property(body, key).get_type() == Variant::NIL);
				}
				CHECK_FALSE(server->soft_body_set_extra_property(body, "skin/pose", Array()));
				CHECK(errors.count() == errors.warning_count());
				CHECK(errors.warning_count() <= 2);
			}
			server->free_rid(body);
		}
		PhysicsServer3D *saved = PhysicsServer3D::get_singleton();
		PhysicsServer3DDummy *dummy = memnew(PhysicsServer3DDummy);
		ErrorCapture errors;
		for (const char *key : REMAINING_CONFIG_KEYS) {
			CHECK_FALSE(dummy->soft_body_set_extra_property(RID(), key, Dictionary()));
			CHECK(dummy->soft_body_get_extra_property(RID(), key).get_type() == Variant::NIL);
		}
		for (const char *key : REMAINING_LIVE_KEYS) {
			CHECK(dummy->soft_body_get_extra_property(RID(), key).get_type() == Variant::NIL);
		}
		CHECK_FALSE(dummy->soft_body_set_extra_property(RID(), "skin/pose", Array()));
		CHECK(dummy->soft_body_get_extra_property_list(RID()).is_empty());
		CHECK(errors.count() == 0);
		memdelete(dummy);
		PhysicsServer3D::set_singleton_for_tests(saved);
	}

	TEST_CASE("remaining P13 invalid rid") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID freed = server->soft_body_create();
		server->free_rid(freed);
		for (RID rid : { RID(), freed }) {
			CAPTURE(rid);
			ErrorCapture errors;
			for (int cap : { 0, 6 }) {
				errors.clear();
				CHECK_FALSE(server->soft_body_set_extra_property(rid, REMAINING_CONFIG_KEYS[cap], remaining_config(cap)));
				CHECK(errors.count() == 1);
			}
			errors.clear();
			CHECK(server->soft_body_get_extra_property(rid, "skin/pose").get_type() == Variant::NIL);
			CHECK(errors.count() == 1);
			errors.clear();
			CHECK(server->soft_body_get_extra_property_list(rid).is_empty());
			CHECK(errors.count() == 1);
		}
	}

	TEST_CASE("remaining P14 registry complete") {
		HashSet<String> sources;
		for (int i = 0; i < Probe::capability_source_file_count(); ++i) {
			sources.insert(Probe::capability_source_file(i));
		}
		HashSet<String> found;
		for (int cap = 0; cap < Probe::capability_count(); ++cap) {
			const String prefix = Probe::capability_prefix(cap);
			const String required = Probe::capability_required_key(cap);
			if (!required.ends_with("/config")) {
				continue;
			}
			CHECK(required == prefix + "config");
			CHECK(sources.has("cap_" + prefix.trim_suffix("/") + ".cpp"));
			found.insert(required);
			bool required_found = false;
			for (int prop = 0; prop < Probe::capability_property_count(cap); ++prop) {
				if (Probe::capability_property_name(cap, prop) == required) {
					required_found = true;
					CHECK_FALSE(Probe::capability_property_live_read(cap, prop));
					CHECK(Probe::capability_property_stored(cap, prop));
					CHECK(Probe::capability_property_clearable(cap, prop));
					CHECK(Probe::capability_property_rebuild(cap, prop));
					CHECK(Probe::capability_property_type(cap, prop) == Variant::DICTIONARY);
				}
			}
			CHECK(required_found);
		}
		CHECK(found.size() == 7);
		for (const char *key : REMAINING_CONFIG_KEYS) {
			CHECK(found.has(key));
		}
	}

	TEST_CASE("remaining P15 composition") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		for (int first = 0; first < 7; ++first) {
			for (int second = 0; second < 7; ++second) {
				if (first == second) {
					continue;
				}
				CAPTURE(first);
				CAPTURE(second);
				const RID body = server->soft_body_create();
				SpaceScope space(server.ptr());
				const RID mesh = make_quad_mesh();
				server->soft_body_set_simulation_precision(body, 20);
				server->soft_body_set_pressure_coefficient(body, 0.0);
				server->soft_body_set_state(body, PS3DE::BODY_STATE_CAN_SLEEP, false);
				server->soft_body_set_mesh(body, mesh);
				const String first_key = REMAINING_CONFIG_KEYS[first];
				const String second_key = REMAINING_CONFIG_KEYS[second];
				const Variant first_value = remaining_config(first);
				const Variant second_value = remaining_config(second);
				REQUIRE(server->soft_body_set_extra_property(body, first_key, first_value));
				server->soft_body_set_space(body, space.rid());
				REQUIRE(Probe::in_space(body));
				const bool replace_first = first == 4 || first == 6, replace_second = second == 4 || second == 6;
				const bool allowed = first == 0 || second == 0 || (!replace_first && !replace_second);
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				CHECK(server->soft_body_set_extra_property(body, second_key, second_value) == allowed);
				if (!allowed) {
					CHECK(errors.has("SBREM-CONFLICT"));
					CHECK(server->soft_body_get_extra_property(body, first_key) == first_value);
					before.unchanged();
				}
				server->free_rid(body);
				RenderingServer::get_singleton()->free_rid(mesh);
			}
		}
	}

	TEST_CASE("remaining P16 order independence") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope first(server.ptr(), space.rid()), second(server.ptr(), space.rid());
		Dictionary vertex;
		vertex["edge_scale"] = 2.0;
		vertex["shear_scale"] = 3.0;
		Dictionary bend;
		bend["compliance"] = 0.01;
		bend["type"] = 1;
		Dictionary lra;
		lra["type"] = 1;
		lra["max_distance_multiplier"] = 1.2;
		const Dictionary configs[] = { vertex, bend, lra };
		const char *keys[] = { "vertex/config", "bend/config", "lra/config" };
		for (RID body : { first.rid(), second.rid() }) {
			server->soft_body_pin_point(body, 0, true);
			for (int i = 0; i < 3; ++i) {
				const int slot = body == first.rid() ? i : 2 - i;
				Dictionary config;
				const Array fields = configs[slot].keys();
				for (int field = fields.size() - 1; field >= 0; --field) {
					config[fields[field]] = configs[slot][fields[field]];
				}
				REQUIRE(server->soft_body_set_extra_property(body, keys[slot], body == first.rid() ? configs[slot] : config));
			}
		}
		CHECK(Probe::edge_count(first.rid()) == Probe::edge_count(second.rid()));
		for (int i = 0; i < Probe::edge_count(first.rid()); ++i) {
			const auto a = Probe::edge(first.rid(), i), b = Probe::edge(second.rid(), i);
			CHECK(a.vertices[0] == b.vertices[0]);
			CHECK(a.vertices[1] == b.vertices[1]);
			CHECK(a.compliance == b.compliance);
			CHECK(a.rest == b.rest);
		}
		CHECK(Probe::dihedral(first.rid(), 0).compliance == Probe::dihedral(second.rid(), 0).compliance);
		for (int i = 0; i < 3; ++i) {
			CHECK(Probe::lra(first.rid(), i).vertices[1] == Probe::lra(second.rid(), i).vertices[1]);
			CHECK(Probe::lra(first.rid(), i).rest == Probe::lra(second.rid(), i).rest);
		}
	}

	TEST_CASE("remaining P17 rebuild policy") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const char *optional[] = { "faces_double_sided", "type", "max_distance_multiplier", "shear_scale", "compliance", "back_stop_radius", "fixed" };
		const Variant defaults[] = { false, 1, 1.0, 1.0, 0.0, 40.0, 0 };
		for (int cap = 0; cap < 7; ++cap) {
			CAPTURE(cap);
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary config = remaining_config(cap);
			config.erase(optional[cap]);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
			if (cap == 5) {
				Array pose;
				pose.push_back(Transform3D(Basis(), Vector3(0, 2, 0)));
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
			before.unchanged();
			Dictionary reversed;
			const Array fields = config.keys();
			for (int i = fields.size() - 1; i >= 0; --i) {
				reversed[fields[i]] = config[fields[i]];
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], reversed));
			before.unchanged();
			const auto generation = Probe::build_counts(body.rid()).generation;
			config[optional[cap]] = defaults[cap];
			REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
			CHECK(Probe::build_counts(body.rid()).generation == generation + 1);
			if (cap == 5) {
				CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == Array(config["initial_pose"]));
			}
			if (cap == 6) {
				config["compliance"] = make_floats(3, 0.5f);
				config["bend"] = make_floats(2, 0.25f);
				config["fixed"] = 2;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "rod/config", config));
				CHECK(Probe::rod_compliance(body.rid(), 0) == 0.5f);
				CHECK(Probe::vertex_inv_mass(body.rid(), 0) == 0);
				Dictionary minimal;
				minimal["joints"] = config["joints"];
				const auto previous = Probe::build_counts(body.rid()).generation;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "rod/config", minimal));
				CHECK(Probe::build_counts(body.rid()).generation == previous + 1);
				CHECK(Probe::rod_compliance(body.rid(), 0) == 0);
				CHECK(Probe::bend_twist_compliance(body.rid(), 0) == 0);
				CHECK(Probe::vertex_inv_mass(body.rid(), 0) > 0);
				CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "rod/config")).size() == 1);
			}
			if (cap == 1 || cap == 2 || cap == 3) {
				const char *field = cap == 1 ? "compliance" : cap == 2 ? "max_distance_multiplier"
																	   : "edge_scale";
				const auto old = Probe::build_counts(body.rid()).generation;
				config[field] = make_floats(1, float(config[field]));
				REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
				CHECK(Probe::build_counts(body.rid()).generation == old + 1);
			}
		}
	}

	TEST_CASE("remaining P18 bind cache") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr()), second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		Dictionary config;
		config["compliance"] = 1e-4;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
		first.step(30);
		PackedVector3Array deformed;
		for (int i = 0; i < 4; ++i) {
			deformed.push_back(server->soft_body_get_point_global_position(body.rid(), i));
		}
		REQUIRE(deformed[0].length() > 1e-3f);
		// Dummy rendering ignores vertex-region uploads. Replace the same RID's
		// surface with measured deformed points so renderer readback changes.
		// Never substitute a fake cache oracle.
		Array replacement_arrays = RenderingServer::get_singleton()->mesh_surface_get_arrays(body.mesh_rid(), 0);
		replacement_arrays[RSE::ARRAY_VERTEX] = deformed;
		RenderingServer::get_singleton()->mesh_clear(body.mesh_rid());
		RenderingServer::get_singleton()->mesh_add_surface_from_arrays(body.mesh_rid(), RSE::PRIMITIVE_TRIANGLES, replacement_arrays);
		const Array updated = RenderingServer::get_singleton()->mesh_surface_get_arrays(body.mesh_rid(), 0);
		CHECK(PackedVector3Array(updated[RSE::ARRAY_VERTEX]) == deformed);
		for (int change = 0; change < 3; ++change) {
			CAPTURE(change);
			if (change == 0) {
				server->soft_body_set_linear_stiffness(body.rid(), 0.25);
			}
			if (change == 1) {
				config["compliance"] = 0.02;
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", config));
			}
			if (change == 2) {
				server->soft_body_set_space(body.rid(), second.rid());
			}
			CHECK(Probe::world_position(body.rid(), 0).length() <= 1e-5f);
			CHECK(Probe::world_position(body.rid(), 1).distance_to(Vector3(1, 0, 0)) <= 1e-5f);
		}
		const auto fixture = fixture_hinge();
		const RID replacement = fixture.create_mesh();
		server->soft_body_set_mesh(body.rid(), replacement);
		CHECK(server->soft_body_get_point_global_position(body.rid(), 3).distance_to(fixture.vertices[3]) <= 1e-5f);
		server->soft_body_set_mesh(body.rid(), body.mesh_rid());
		RenderingServer::get_singleton()->free_rid(replacement);
	}

	TEST_CASE("remaining P19 late context") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const RID body = server->soft_body_create();
		const Dictionary config = fixture_skin();
		REQUIRE(server->soft_body_set_extra_property(body, "skin/config", config));
		CHECK_FALSE(Probe::in_space(body));
		server->soft_body_set_space(body, space.rid());
		CHECK_FALSE(Probe::in_space(body));
		RemainingMeshFixture incompatible;
		for (const Vector3 &point : { Vector3(), Vector3(1, 0, 0), Vector3(0, 0, 1) }) {
			incompatible.vertices.push_back(point);
		}
		for (int i : { 0, 1, 2 }) {
			incompatible.indices.push_back(i);
		}
		const RID bad = incompatible.create_mesh();
		{
			ErrorCapture errors;
			server->soft_body_set_mesh(body, bad);
			CHECK(errors.has("SBREM-INDEX"));
			CHECK_FALSE(Probe::in_space(body));
			CHECK(Dictionary(server->soft_body_get_extra_property(body, "skin/config")) == config);
			CHECK(Probe::build_counts(body).generation == 0);
		}
		const RID good = make_quad_mesh();
		server->soft_body_set_mesh(body, good);
		CHECK(Probe::in_space(body));
		CHECK(Probe::skinned_count(body) == 4);
		CHECK(Probe::build_counts(body).generation == 1);
		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(bad);
		RenderingServer::get_singleton()->free_rid(good);
	}

	TEST_CASE("remaining P21 override removal") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		server->soft_body_set_space(body.rid(), RID());
		server->soft_body_set_space(body.rid(), space.rid());
		const Dictionary defaults = server->soft_body_get_extra_property(body.rid(), "scalar/current");
		Dictionary full;
		full["friction"] = 0.2;
		full["restitution"] = 0.8;
		full["gravity_factor"] = 2.0;
		full["vertex_radius"] = 0.2;
		full["faces_double_sided"] = true;
		for (const Variant *field = full.next(nullptr); field != nullptr; field = full.next(field)) {
			CAPTURE(String(*field));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", full));
			Dictionary reduced = full.duplicate(true);
			reduced.erase(*field);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", reduced));
			const Dictionary actual = server->soft_body_get_extra_property(body.rid(), "scalar/current");
			for (const Variant *other = full.next(nullptr); other != nullptr; other = full.next(other)) {
				const Variant expected = *other == *field ? defaults[*other] : full[*other];
				if (expected.get_type() == Variant::BOOL) {
					CHECK(bool(actual[*other]) == bool(expected));
				} else {
					CHECK(double(actual[*other]) == doctest::Approx(double(expected)).epsilon(1e-5));
				}
			}
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", Variant()));
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/current")) == defaults);
		}
	}

	TEST_CASE("remaining P20 thread guard") {
		{
			Probe::ThreadedServerScope owner;
			auto *server = static_cast<PhysicsServer3D *>(owner.server());
			REQUIRE(owner.actual_separate_thread());
			const Variant old_setting = GLOBAL_GET("physics/3d/run_on_separate_thread");
			const RID mesh = make_quad_mesh();
			for (bool setting : { false, true }) {
				CAPTURE(setting);
				ProjectSettings::get_singleton()->set("physics/3d/run_on_separate_thread", setting);
				for (int cap : { 1, 2, 3, 5 }) {
					CAPTURE(cap);
					const RID body = server->soft_body_create();
					RemainingSnapshot before(server, body);
					ErrorCapture errors;
					CHECK_FALSE(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[cap], remaining_config(cap)));
					CHECK(errors.has("SBREM-THREAD"));
					CHECK(Probe::build_counts(body).renderer_reads == 0);
					before.unchanged();
					server->free_rid(body);
				}
				const RID body = server->soft_body_create();
				server->soft_body_set_mesh(body, mesh);
				server->soft_body_get_extra_property_list(body);
				{
					RemainingSnapshot before(server, body);
					ErrorCapture errors;
					CHECK_FALSE(server->soft_body_set_extra_property(body, "scalar/config", remaining_config(0)));
					CHECK(errors.has("SBREM-THREAD"));
					CHECK(Probe::build_counts(body).renderer_reads == 0);
					before.unchanged();
				}
				server->free_rid(body);
			}
			ProjectSettings::get_singleton()->set("physics/3d/run_on_separate_thread", old_setting);
			for (int replacement : { 4, 6 }) {
				for (bool scalar_neighbor : { false, true }) {
					CAPTURE(replacement);
					CAPTURE(scalar_neighbor);
					SpaceScope space(server);
					const RID body = server->soft_body_create();
					REQUIRE(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[replacement], remaining_config(replacement)));
					server->soft_body_set_mesh(body, mesh);
					server->soft_body_set_space(body, space.rid());
					// Synchronous public read drains queued mesh/space writes before probes.
					server->soft_body_get_extra_property_list(body);
					CHECK(Probe::in_space(body));
					space.step(2);
					if (replacement == 6) {
						const PackedFloat32Array state = server->soft_body_get_extra_property(body, "rod/state");
						REQUIRE(state.size() == 12);
						for (float value : state) {
							CHECK(Math::is_finite(value));
						}
					} else {
						CHECK(int(server->soft_body_get_extra_property(body, "volume/count")) == 1);
					}
					if (scalar_neighbor) {
						REQUIRE(server->soft_body_set_extra_property(body, "scalar/config", remaining_config(0)));
					}
					RemainingSnapshot before(server, body);
					ErrorCapture errors;
					CHECK_FALSE(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[replacement], Variant()));
					CHECK(errors.has("SBREM-THREAD"));
					before.unchanged();
					CHECK(Probe::build_counts(body).renderer_reads == 0);
					server->free_rid(body);
				}
			}
			{
				SpaceScope space(server);
				const RID body = server->soft_body_create();
				const Dictionary scalar = remaining_config(0);
				REQUIRE(server->soft_body_set_extra_property(body, "scalar/config", scalar));
				ErrorCapture errors;
				server->soft_body_set_mesh(body, mesh);
				CHECK(Dictionary(server->soft_body_get_extra_property(body, "scalar/config")) == scalar);
				CHECK(errors.has("SBREM-THREAD"));
				CHECK_FALSE(Probe::in_space(body));
				errors.clear();
				server->soft_body_set_space(body, space.rid());
				CHECK(Dictionary(server->soft_body_get_extra_property(body, "scalar/config")) == scalar);
				CHECK(errors.has("SBREM-THREAD"));
				CHECK_FALSE(Probe::in_space(body));
				CHECK(Probe::build_counts(body).renderer_reads == 0);
				CHECK(Probe::build_counts(body).generation == 0);
				errors.clear();
				REQUIRE(server->soft_body_set_extra_property(body, "volume/config", fixture_tetra()));
				CHECK(int(server->soft_body_get_extra_property(body, "volume/count")) == 1);
				CHECK(Probe::build_counts(body).generation == 1);
				CHECK(Probe::build_counts(body).renderer_reads == 0);
				server->free_rid(body);
			}
			RenderingServer::get_singleton()->free_rid(mesh);
		}
		// Real WrapMT transport, deliberately not a threaded skin solver claim.
		class RecordingBackend : public PhysicsServer3DDummy {
		public:
			Array snapshots;
			Vector<RID> ids;
			Vector<StringName> keys;
			bool soft_body_set_extra_property(RID p_body, const StringName &p_key, const Variant &p_value) override {
				ids.push_back(p_body);
				keys.push_back(p_key);
				snapshots.push_back(p_value.duplicate(true));
				return snapshots.size() == 1;
			}
		};
		PhysicsServer3D *saved = PhysicsServer3D::get_singleton();
		auto *recording = memnew(RecordingBackend);
		auto *wrapper = memnew(PhysicsServer3DWrapMT(recording, true));
		wrapper->init();
		const RID rid = RID::from_uint64(1234);
		for (int frame = 0; frame < 2; ++frame) {
			Array pose;
			pose.push_back(Transform3D(Basis(Vector3(0, 1, 0), 0.2f + frame).scaled(Vector3(2, 3, 4)), Vector3(frame, 2, 3)));
			pose.push_back(Transform3D(Basis(Vector3(0, 0, 1), -0.3f - frame).scaled(Vector3(0.5f, 1, 2)), Vector3(4, 5, frame)));
			const Array expected = pose.duplicate(true);
			CHECK(wrapper->soft_body_set_extra_property(rid, "skin/pose", pose) == (frame == 0));
			pose.clear();
			CHECK(recording->ids[frame] == rid);
			CHECK(recording->keys[frame] == StringName("skin/pose"));
			CHECK(Array(recording->snapshots[frame]) == expected);
		}
		wrapper->finish();
		memdelete(wrapper);
		PhysicsServer3D::set_singleton_for_tests(saved);
	}

	TEST_CASE("remaining R07 threaded value path") {
		Probe::ThreadedServerScope owner;
		auto *server = static_cast<PhysicsServer3D *>(owner.server());
		REQUIRE(owner.actual_separate_thread());
		SpaceScope space(server);
		Dictionary original = fixture_tetra();
		const Dictionary expected = original.duplicate(true);
		const RID body = server->soft_body_create();
		server->soft_body_set_simulation_precision(body, 20);
		server->soft_body_set_total_mass(body, 1.0);
		server->soft_body_set_state(body, PS3DE::BODY_STATE_CAN_SLEEP, false);
		REQUIRE(server->soft_body_set_extra_property(body, "volume/config", original));
		original.clear();
		Dictionary scalar;
		scalar["gravity_factor"] = 2.0;
		REQUIRE(server->soft_body_set_extra_property(body, "scalar/config", scalar));
		scalar.clear();
		server->soft_body_set_space(body, space.rid());
		CHECK(Dictionary(server->soft_body_get_extra_property(body, "volume/config")) == expected);
		CHECK_FALSE(find_property(server->soft_body_get_extra_property_list(body), "volume/positions").is_empty());
		const PackedVector3Array before = server->soft_body_get_extra_property(body, "volume/positions");
		REQUIRE(before.size() == 4);
		space.step(30);
		server->sync();
		server->end_sync();
		const PackedVector3Array after = server->soft_body_get_extra_property(body, "volume/positions");
		REQUIRE(after.size() == 4);
		CHECK(after[0].y < before[0].y - 1e-3f);
		CHECK(double(Dictionary(server->soft_body_get_extra_property(body, "scalar/current"))["gravity_factor"]) == 2.0);
		CHECK(Probe::all_finite(body));
		CHECK(Probe::build_counts(body).renderer_reads == 0);
		server->free_rid(body);
	}

	TEST_CASE("remaining A09 base validity") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SUBCASE("negative pin is geometry independent before first mesh") {
			for (bool configured : { false, true }) {
				CAPTURE(configured);
				const RID body = server->soft_body_create();
				const Dictionary scalar = remaining_config(0);
				if (configured) {
					REQUIRE(server->soft_body_set_extra_property(body, "scalar/config", scalar));
				}
				for (int i = 0; i < Probe::vertex_count(body); ++i) {
					REQUIRE(Probe::set_vertex_local(body, i, Probe::vertex_position(body, i), Vector3(0.25f, -0.5f, 0.75f)));
					CHECK(Probe::velocity(body, i).length() > 0.1f);
				}
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				server->soft_body_pin_point(body, -1, true);
				if (configured) {
					CHECK(errors.has("SBREM-INDEX"));
					CHECK(errors.has("typed.pin"));
					CHECK(errors.has("-1"));
					CHECK_FALSE(server->soft_body_is_point_pinned(body, -1));
					before.unchanged();
				} else {
					CHECK(errors.count() == 0); // Stock behavior stays unchanged.
					// Stock getter requires space; candidate config exposes saved pins.
					CHECK_FALSE(server->soft_body_set_extra_property(body, "scalar/config", scalar));
					CHECK(errors.has("SBREM-INDEX"));
					before.unchanged();
					server->soft_body_pin_point(body, -1, false);
					REQUIRE(server->soft_body_set_extra_property(body, "scalar/config", scalar));
				}
				errors.clear();
				server->soft_body_pin_point(body, 0, true);
				CHECK(errors.count() == 0); // Upper bound awaits geometry.
				const RID mesh = make_quad_mesh();
				SpaceScope space(server.ptr());
				server->soft_body_set_mesh(body, mesh);
				server->soft_body_set_space(body, space.rid());
				CHECK(errors.count() == 0);
				REQUIRE(Probe::in_space(body));
				CHECK(Probe::shared_inv_mass(body, 0) == 0);
				CHECK(Probe::vertex_inv_mass(body, 0) == 0);
				CHECK(Dictionary(server->soft_body_get_extra_property(body, "scalar/config")) == scalar);
				server->free_rid(body);
				RenderingServer::get_singleton()->free_rid(mesh);
			}
		}

		for (int path = 0; path < 8; ++path) {
			CAPTURE(path);
			SpaceScope space(server.ptr());
			const RID body = server->soft_body_create();
			RID mesh;
			server->soft_body_set_total_mass(body, 1.0);
			server->soft_body_set_simulation_precision(body, 20);
			if (path < 5) {
				mesh = make_quad_mesh();
				server->soft_body_set_mesh(body, mesh);
			}
			const int cap = path < 4 ? path : path == 4 ? 5
					: path == 5							? 4
					: path == 6							? 6
														: 0;
			REQUIRE(server->soft_body_set_extra_property(body, REMAINING_CONFIG_KEYS[cap], remaining_config(cap)));
			server->soft_body_set_space(body, space.rid());
			for (float value : { 0.0f, -1.0f, NAN, INFINITY, -INFINITY, std::numeric_limits<float>::denorm_min(), FLT_MAX }) {
				CAPTURE(value);
				for (int i = 0; i < Probe::vertex_count(body); ++i) {
					REQUIRE(Probe::set_vertex_local(body, i, Probe::vertex_position(body, i), Vector3(0.25f, -0.5f, 0.75f)));
					CHECK(Probe::velocity(body, i).length() > 0.1f);
				}
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				const bool valid = std::isfinite(value) && value > 0 && (path == 7 || value > std::numeric_limits<float>::denorm_min());
				server->soft_body_set_total_mass(body, value);
				if (!valid) {
					CHECK(errors.has("SBREM-CONTEXT"));
					before.unchanged();
				} else {
					CHECK(server->soft_body_get_total_mass(body) == value);
					if (path != 7) {
						for (int i = 0; i < Probe::vertex_count(body); ++i) {
							const float inverse = Probe::vertex_inv_mass(body, i);
							CHECK(Math::is_finite(inverse));
							if (inverse != 0 || path != 6 || i != 0) {
								CHECK(inverse > 0);
							}
						}
					}
				}
				server->soft_body_set_total_mass(body, 1.0);
			}
			for (int value : { 0, -1, 1, INT32_MAX }) {
				CAPTURE(value);
				for (int i = 0; i < Probe::vertex_count(body); ++i) {
					REQUIRE(Probe::set_vertex_local(body, i, Probe::vertex_position(body, i), Vector3(0.25f, -0.5f, 0.75f)));
					CHECK(Probe::velocity(body, i).length() > 0.1f);
				}
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				server->soft_body_set_simulation_precision(body, value);
				if (value < 1) {
					CHECK(errors.has("SBREM-CONTEXT"));
					before.unchanged();
				} else {
					CHECK(server->soft_body_get_simulation_precision(body) == value);
				}
				server->soft_body_set_simulation_precision(body, 20);
			}
			for (float value : { 0.0f, -1.0f, 1.0f, 1.01f, NAN, INFINITY, -INFINITY, std::numeric_limits<float>::denorm_min() }) {
				CAPTURE(value);
				for (int i = 0; i < Probe::vertex_count(body); ++i) {
					REQUIRE(Probe::set_vertex_local(body, i, Probe::vertex_position(body, i), Vector3(0.25f, -0.5f, 0.75f)));
					CHECK(Probe::velocity(body, i).length() > 0.1f);
				}
				RemainingSnapshot before(server.ptr(), body);
				ErrorCapture errors;
				const bool valid = std::isfinite(value) && value >= 0 && value <= 1 && (path >= 5 || (value > 0 && value != std::numeric_limits<float>::denorm_min()));
				server->soft_body_set_linear_stiffness(body, value);
				if (!valid) {
					CHECK(errors.has("SBREM-CONTEXT"));
					before.unchanged();
				} else {
					CHECK(server->soft_body_get_linear_stiffness(body) == value);
				}
				server->soft_body_set_linear_stiffness(body, 0.5);
			}
			server->free_rid(body);
			if (mesh.is_valid()) {
				RenderingServer::get_singleton()->free_rid(mesh);
			}
		}
		// A positive widened input can underflow float32, but public float32 mass
		// cannot express the inverse mass which would cause this defense to fire.
		float converted = 1;
		String error;
		CHECK_FALSE(SoftBodyCapValidation::finite_float(std::numeric_limits<double>::min(), converted, &error, "inverse_mass", true));
		CHECK(error.contains("positive"));
		CHECK(SoftBodyCapValidation::finite_float(4.0 / FLT_MAX, converted, &error, "inverse_mass", true));
		CHECK(converted > 0);
		{
			SpaceScope space(server.ptr());
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary vertex;
			vertex["edge_scale"] = double(FLT_MAX);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", vertex));
			for (int i = 0; i < Probe::vertex_count(body.rid()); ++i) {
				REQUIRE(Probe::set_vertex_local(body.rid(), i, Probe::vertex_position(body.rid(), i), Vector3(0.25f, -0.5f, 0.75f)));
				CHECK(Probe::velocity(body.rid(), i).length() > 0.1f);
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			server->soft_body_set_total_mass(body.rid(), 1e-6f);
			CHECK(errors.has("SBREM-CONTEXT"));
			before.unchanged();
		}
		{
			SpaceScope space(server.ptr());
			Dictionary config = fixture_tetra();
			PackedInt32Array fixed;
			for (int i = 0; i < 4; ++i) {
				fixed.push_back(i);
			}
			config["fixed"] = fixed;
			VolumeBodyScope body(server.ptr(), space.rid(), config);
			for (int i = 0; i < Probe::vertex_count(body.rid()); ++i) {
				REQUIRE(Probe::set_vertex_local(body.rid(), i, Probe::vertex_position(body.rid(), i), Vector3(0.25f, -0.5f, 0.75f)));
				CHECK(Probe::velocity(body.rid(), i).length() > 0.1f);
			}
			RemainingSnapshot before(server.ptr(), body.rid());
			ErrorCapture errors;
			server->soft_body_set_total_mass(body.rid(), std::numeric_limits<float>::denorm_min());
			CHECK(errors.has("SBREM-CONTEXT"));
			before.unchanged();
		}
		for (int kind = 0; kind < 4; ++kind) {
			CAPTURE(kind);
			const RID body = server->soft_body_create();
			const RID mesh = make_quad_mesh();
			server->soft_body_set_mesh(body, mesh);
			if (kind == 0) {
				server->soft_body_set_simulation_precision(body, 0);
			}
			if (kind == 1) {
				server->soft_body_set_linear_stiffness(body, 0);
			}
			if (kind == 2) {
				server->soft_body_set_total_mass(body, NAN);
			}
			if (kind == 3) {
				server->soft_body_set_total_mass(body, std::numeric_limits<float>::denorm_min());
			}
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body, "scalar/config", remaining_config(0)));
			CHECK(errors.has("SBREM-CONTEXT"));
			CHECK(server->soft_body_get_extra_property(body, "scalar/config").get_type() == Variant::NIL);
			CHECK_FALSE(Probe::in_space(body));
			CHECK(Probe::build_counts(body).generation == 0);
			if (kind == 2) {
				CHECK(Math::is_nan(server->soft_body_get_total_mass(body)));
			}
			server->free_rid(body);
			RenderingServer::get_singleton()->free_rid(mesh);
		}
		for (bool rod : { false, true }) {
			const RID body = server->soft_body_create();
			if (rod) {
				REQUIRE(set_valid_rod(server.ptr(), body, 4));
			}
			RemainingSnapshot before(server.ptr(), body);
			ErrorCapture errors;
			server->soft_body_set_simulation_precision(body, 0);
			server->soft_body_set_linear_stiffness(body, -1.0);
			if (rod) {
				CHECK(errors.has("SBREM-CONTEXT"));
				before.unchanged();
			} else {
				CHECK_FALSE(errors.has("SBREM-CONTEXT"));
				CHECK(server->soft_body_get_simulation_precision(body) == 0);
				CHECK(server->soft_body_get_linear_stiffness(body) == 0);
			}
			server->free_rid(body);
		}
	}

	TEST_CASE("remaining R01 diagnostic inventory") {
		// Real production diagnostics, never DEBUG_ENABLED-only asserts.
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_seam();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		HashSet<String> observed;
		const char *codes[] = { "SBREM-KEY", "SBREM-TYPE", "SBREM-VALUE", "SBREM-SIZE", "SBREM-INDEX", "SBREM-ALIAS", "SBREM-TOPOLOGY", "SBREM-CONFLICT", "SBREM-CONTEXT", "SBREM-POSE", "SBREM-SKIN-NORMAL", "SBREM-LRA-NO-ANCHOR" };
		for (int kind = 0; kind < 12; ++kind) {
			CAPTURE(codes[kind]);
			ErrorCapture errors;
			if (kind == 0) {
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "scalar/wrong", 0));
			}
			if (kind == 1) {
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "scalar/config", true));
			}
			if (kind == 2) {
				Dictionary bad;
				bad["friction"] = -1.0;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "scalar/config", bad));
			}
			if (kind == 3) {
				Dictionary bad;
				bad["edge_scale"] = PackedFloat32Array();
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "vertex/config", bad));
			}
			if (kind == 4) {
				Dictionary bad = fixture_tetra();
				PackedInt32Array fixed;
				fixed.push_back(8);
				bad["fixed"] = fixed;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
			}
			if (kind == 5) {
				Dictionary bad;
				PackedFloat32Array values = make_floats(7, 1);
				values.set(4, 2);
				bad["edge_scale"] = values;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "vertex/config", bad));
			}
			if (kind == 6) {
				Dictionary bad = fixture_tetra();
				PackedVector3Array points = bad["vertices"];
				points.set(3, Vector3());
				bad["vertices"] = points;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", bad));
			}
			if (kind == 7) {
				server->soft_body_pin_point(body.rid(), 0, true);
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "volume/config", fixture_tetra()));
				server->soft_body_pin_point(body.rid(), 0, false);
			}
			if (kind == 8) {
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", remaining_config(0)));
				server->soft_body_set_total_mass(body.rid(), -1.0);
			}
			if (kind == 9) {
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/pose", Array()));
			}
			if (kind == 10) {
				Dictionary skin = fixture_skin(1, 2);
				PackedInt32Array selected;
				selected.push_back(0);
				selected.push_back(4);
				skin["vertices"] = selected;
				skin["back_stop_distance"] = 0.0;
				CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
			}
			if (kind == 11) {
				REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", remaining_config(2)));
			}
			CHECK(errors.has(codes[kind]));
			CHECK(errors.joined().length() > String(codes[kind]).length() + 10);
			if (errors.has(codes[kind])) {
				observed.insert(codes[kind]);
			}
		}
		CHECK(observed.size() == 12);
		{
			Probe::ThreadedServerScope owner;
			auto *threaded = static_cast<PhysicsServer3D *>(owner.server());
			const RID rid = threaded->soft_body_create();
			ErrorCapture errors;
			CHECK_FALSE(threaded->soft_body_set_extra_property(rid, "skin/config", fixture_skin()));
			CHECK(errors.has("SBREM-THREAD"));
			threaded->free_rid(rid);
		}
	}

	TEST_CASE("remaining R02 rid isolation") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool built : { false, true }) {
			CAPTURE(built);
			const RID mesh = make_quad_mesh();
			const RID old = server->soft_body_create();
			REQUIRE(server->soft_body_set_extra_property(old, "skin/config", fixture_skin()));
			Array pose;
			pose.push_back(Transform3D(Basis(), Vector3(0, 3, 0)));
			REQUIRE(server->soft_body_set_extra_property(old, "skin/pose", pose));
			if (built) {
				server->soft_body_set_mesh(old, mesh);
				server->soft_body_set_space(old, space.rid());
				space.step();
				REQUIRE(Probe::in_space(old));
			}
			server->free_rid(old);
			const RID fresh = server->soft_body_create();
			for (const char *key : REMAINING_CONFIG_KEYS) {
				CHECK(server->soft_body_get_extra_property(fresh, key).get_type() == Variant::NIL);
			}
			CHECK(server->soft_body_get_extra_property(fresh, "skin/pose").get_type() == Variant::NIL);
			CHECK(Probe::build_counts(fresh).generation == 0);
			CHECK(Probe::body_identity(fresh) == UINT64_MAX);
			server->free_rid(fresh);
			RenderingServer::get_singleton()->free_rid(mesh);
		}
	}

	TEST_CASE("remaining R03 mesh recovery") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope first(server.ptr()), second(server.ptr());
		MeshBodyScope body(server.ptr(), first.rid());
		Dictionary config = fixture_skin();
		config["max_distance"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", config));
		Array pose;
		pose.push_back(Transform3D(Basis(), Vector3(0, 2, 0)));
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/pose", pose));
		first.step();
		const auto generation = Probe::build_counts(body.rid()).generation;
		RemainingMeshFixture bad_fixture;
		for (const Vector3 &point : { Vector3(), Vector3(1, 0, 0), Vector3(0, 0, 1) }) {
			bad_fixture.vertices.push_back(point);
		}
		for (int i : { 0, 1, 2 }) {
			bad_fixture.indices.push_back(i);
		}
		const RID bad_mesh = bad_fixture.create_mesh();
		{
			ErrorCapture errors;
			server->soft_body_set_mesh(body.rid(), bad_mesh);
			CHECK(errors.has("SBREM-INDEX"));
			CHECK_FALSE(Probe::in_space(body.rid()));
			CHECK(Probe::body_identity(body.rid()) == UINT64_MAX);
			CHECK(Probe::build_counts(body.rid()).generation == generation);
			CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "skin/config")) == config);
			CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == pose);
			errors.clear();
			server->soft_body_set_space(body.rid(), second.rid());
			CHECK(errors.has("SBREM-INDEX"));
			CHECK(server->soft_body_get_space(body.rid()) == second.rid());
		}
		for (const char *key : REMAINING_LIVE_KEYS) {
			ErrorCapture errors;
			CHECK(server->soft_body_get_extra_property(body.rid(), key).get_type() == Variant::NIL);
			CHECK(errors.count() == 1);
		}
		server->soft_body_set_mesh(body.rid(), body.mesh_rid());
		CHECK(Probe::in_space(body.rid()));
		CHECK(Probe::build_counts(body.rid()).generation == generation + 1);
		second.step();
		CHECK(Probe::world_position(body.rid(), 0).y == doctest::Approx(2.0).epsilon(1e-4));
		{
			ErrorCapture errors;
			server->soft_body_set_space(body.rid(), RID());
			CHECK(errors.count() == 0);
			CHECK_FALSE(Probe::in_space(body.rid()));
			CHECK(Probe::build_counts(body.rid()).generation == generation + 1);
			CHECK(Array(server->soft_body_get_extra_property(body.rid(), "skin/pose")) == pose);
		}
		RenderingServer::get_singleton()->free_rid(bad_mesh);
	}

	TEST_CASE("remaining R04 full mesh stack") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool hard : { false, true }) {
			CAPTURE(hard);
			MeshBodyScope body(server.ptr(), space.rid());
			server->soft_body_pin_point(body.rid(), 1, true);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", remaining_config(0)));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", remaining_config(1)));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "lra/config", remaining_config(2)));
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", remaining_config(3)));
			Dictionary skin = fixture_skin(1, 2);
			PackedInt32Array vertices = skin["vertices"];
			vertices.set(1, 2);
			skin["vertices"] = vertices;
			skin["max_distance"] = hard ? 0.0 : 0.05;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
			CHECK(Probe::dihedral_bend_count(body.rid()) == 1);
			CHECK(Probe::skinned_count(body.rid()) == 2);
			CHECK(Probe::lra_count(body.rid()) == (hard ? 1 : 3));
			CHECK(Probe::scalars(body.rid()).friction == doctest::Approx(0.4).epsilon(1e-5));
			CHECK(Probe::vertex_inv_mass(body.rid(), 1) == 0);
			for (int i : { 0, 2 }) {
				CHECK(Probe::vertex_inv_mass(body.rid(), i) == (hard ? 0 : 4));
			}
			space.step(30);
			CHECK(Probe::all_finite(body.rid()));
		}
	}

	TEST_CASE("remaining R05 body isolation") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope first(server.ptr(), space.rid()), second(server.ptr(), space.rid());
		server->soft_body_set_mesh(second.rid(), first.mesh_rid());
		Dictionary a = fixture_skin();
		a["max_distance"] = 0.0;
		Dictionary b = a.duplicate(true);
		Array pose;
		pose.push_back(Transform3D(Basis(), Vector3(0, 2, 0)));
		b["initial_pose"] = pose;
		REQUIRE(server->soft_body_set_extra_property(first.rid(), "skin/config", a));
		REQUIRE(server->soft_body_set_extra_property(second.rid(), "skin/config", b));
		space.step();
		CHECK(Probe::settings_identity(first.rid()) != Probe::settings_identity(second.rid()));
		RemainingSnapshot before(server.ptr(), second.rid());
		server->soft_body_set_total_mass(first.rid(), 2.0);
		before.unchanged();
		pose[0] = Transform3D(Basis(), Vector3(0, 4, 0));
		REQUIRE(server->soft_body_set_extra_property(first.rid(), "skin/pose", pose));
		before.unchanged();
		CHECK(Probe::world_position(second.rid(), 0).y == doctest::Approx(2));
	}

	TEST_CASE("remaining R06 family teardown") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool reverse : { false, true }) {
			CAPTURE(reverse);
			MeshBodyScope body(server.ptr(), space.rid());
			server->soft_body_pin_point(body.rid(), 0, true);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", remaining_config(0)));
			const int caps[] = { 1, 2, 3, 5 };
			for (int cap : caps) {
				REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], remaining_config(cap)));
			}
			for (int i = 0; i < 4; ++i) {
				REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[caps[reverse ? 3 - i : i]], Variant()));
			}
			CHECK(Probe::dihedral_bend_count(body.rid()) == 0);
			CHECK(Probe::lra_count(body.rid()) == 0);
			CHECK(Probe::skinned_count(body.rid()) == 0);
			CHECK(server->soft_body_get_extra_property(body.rid(), "skin/pose").get_type() == Variant::NIL);
			Array arrays = RenderingServer::get_singleton()->mesh_surface_get_arrays(body.mesh_rid(), 0);
			PackedVector3Array points = arrays[RSE::ARRAY_VERTEX];
			for (int i = 0; i < points.size(); ++i) {
				points.set(i, points[i] + Vector3(5, 0, 0));
			}
			arrays[RSE::ARRAY_VERTEX] = points;
			RenderingServer::get_singleton()->mesh_clear(body.mesh_rid());
			RenderingServer::get_singleton()->mesh_add_surface_from_arrays(body.mesh_rid(), RSE::PRIMITIVE_TRIANGLES, arrays);
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", remaining_config(1)));
			CHECK(Probe::world_position(body.rid(), 0).distance_to(Vector3(5, 0, 0)) <= 1e-5f);
		}
	}

	TEST_CASE("remaining R08 prestep ordering") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		{
			SpaceScope space(server.ptr());
			MeshBodyScope body(server.ptr(), space.rid());
			Dictionary skin = fixture_skin();
			skin["max_distance"] = 0.0;
			REQUIRE(server->soft_body_set_extra_property(body.rid(), "skin/config", skin));
			space.step(180);
			const auto counts = Probe::build_counts(body.rid());
			CHECK(counts.pre_step == 180);
			CHECK(counts.solver_steps == 180);
			CHECK(counts.skin_init == 1);
			CHECK(counts.skin_recurring == 179);
			server->soft_body_set_space(body.rid(), RID());
			server->soft_body_set_space(body.rid(), space.rid());
			space.step();
			CHECK(Probe::build_counts(body.rid()).skin_init == 2);
			CHECK(Probe::build_counts(body.rid()).solver_steps == 181);
		}
		for (bool initialized : { false, true }) {
			for (int recovery = 0; recovery < 4; ++recovery) {
				CAPTURE(initialized);
				CAPTURE(recovery);
				SpaceScope space(server.ptr());
				MeshBodyScope left(server.ptr(), space.rid()), fault(server.ptr(), space.rid()), right(server.ptr(), space.rid());
				Dictionary skin = fixture_skin();
				skin["max_distance"] = 0.0;
				for (RID body : { left.rid(), right.rid() }) {
					REQUIRE(server->soft_body_set_extra_property(body, "skin/config", skin));
				}
				skin = fixture_skin(1, 1);
				skin["max_distance"] = 0.0;
				REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/config", skin));
				if (initialized) {
					space.step();
				}
				Array arrays = RenderingServer::get_singleton()->mesh_surface_get_arrays(fault.mesh_rid(), 0);
				PackedVector3Array altered = arrays[RSE::ARRAY_VERTEX];
				for (int i = 0; i < altered.size(); ++i) {
					altered.set(i, altered[i] + Vector3(5, 0, 0));
				}
				arrays[RSE::ARRAY_VERTEX] = altered;
				RenderingServer::get_singleton()->mesh_clear(fault.mesh_rid());
				RenderingServer::get_singleton()->mesh_add_surface_from_arrays(fault.mesh_rid(), RSE::PRIMITIVE_TRIANGLES, arrays);
				Array far;
				far.push_back(Transform3D(Basis(Vector3(2.5e38f, 2.5e38f, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3()));
				REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/pose", far));
				REQUIRE(Probe::set_body_frame(fault.rid(), Transform3D(Basis(Vector3(0, 0, 1), Math::PI / 4), Vector3(3, 4, 5))));
				const auto before = Probe::build_counts(fault.rid()), healthy = Probe::build_counts(left.rid());
				ErrorCapture errors;
				space.step();
				CAPTURE(errors.joined());
				CHECK(errors.count() == 1);
				CHECK(errors.has("SBREM-POSE"));
				CHECK(errors.has("pre_step"));
				CHECK_FALSE(Probe::in_space(fault.rid()));
				const auto after = Probe::build_counts(fault.rid());
				CHECK(after.generation == before.generation);
				CHECK(after.removals_during_traversal == 0);
				CHECK(after.solver_steps == before.solver_steps);
				CHECK(after.skin_init == before.skin_init);
				CHECK(after.skin_recurring == before.skin_recurring);
				CHECK(after.pre_step == before.pre_step + 1);
				CHECK(Probe::build_counts(left.rid()).solver_steps == healthy.solver_steps + 1);
				CHECK(Probe::build_counts(right.rid()).solver_steps == healthy.solver_steps + 1);
				errors.clear();
				Array bad;
				bad.push_back(true);
				CHECK_FALSE(server->soft_body_set_extra_property(fault.rid(), "skin/pose", bad));
				CHECK(errors.has("SBREM-TYPE"));
				CHECK(Array(server->soft_body_get_extra_property(fault.rid(), "skin/pose")) == far);
				Array latest;
				latest.push_back(Transform3D(Basis(), Vector3(0, 3, 0)));
				REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/pose", latest));
				REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/config", skin));
				CHECK_FALSE(Probe::in_space(fault.rid()));
				Dictionary bad_structure;
				bad_structure["edge_scale"] = make_floats(5, 1);
				errors.clear();
				CHECK_FALSE(server->soft_body_set_extra_property(fault.rid(), "vertex/config", bad_structure));
				CHECK(errors.has("SBREM-SIZE"));
				CHECK_FALSE(Probe::in_space(fault.rid()));
				errors.clear();
				space.step(3);
				CHECK(errors.count() == 0);
				CHECK(Probe::build_counts(fault.rid()).generation == before.generation);
				RID new_mesh;
				Vector3 expected(0, 3, 0);
				if (recovery == 0) {
					server->soft_body_set_space(fault.rid(), RID());
					server->soft_body_set_space(fault.rid(), space.rid());
				}
				if (recovery == 1) {
					REQUIRE(server->soft_body_set_extra_property(fault.rid(), "scalar/config", remaining_config(0)));
				}
				if (recovery == 2) {
					Dictionary changed = skin.duplicate(true);
					Array initial;
					initial.push_back(Transform3D(Basis(), Vector3(0, 2, 0)));
					changed["initial_pose"] = initial;
					REQUIRE(server->soft_body_set_extra_property(fault.rid(), "skin/config", changed));
					expected.y = 2;
				}
				if (recovery == 3) {
					auto fixture = fixture_hinge();
					for (int i = 0; i < fixture.vertices.size(); ++i) {
						fixture.vertices.set(i, fixture.vertices[i] + Vector3(2, 0, 0));
					}
					new_mesh = fixture.create_mesh();
					server->soft_body_set_mesh(fault.rid(), new_mesh);
					expected.x = 2;
				}
				REQUIRE(Probe::in_space(fault.rid()));
				CHECK(Probe::build_counts(fault.rid()).generation == before.generation + 1);
				space.step();
				CHECK(Probe::build_counts(fault.rid()).skin_init == before.skin_init + 1);
				CHECK(Probe::world_position(fault.rid(), 0).distance_to(expected) <= 1e-4f);
				CHECK(Probe::all_finite(left.rid()));
				CHECK(Probe::all_finite(right.rid()));
				if (new_mesh.is_valid()) {
					server->soft_body_set_mesh(fault.rid(), fault.mesh_rid());
					RenderingServer::get_singleton()->free_rid(new_mesh);
				}
			}
		}
		for (bool initialized : { false, true }) {
			CAPTURE(initialized);
			SpaceScope space(server.ptr());
			MeshBodyScope healthy(server.ptr(), space.rid());
			const RID mesh = make_quad_mesh();
			const RID fault = server->soft_body_create();
			server->soft_body_set_mesh(fault, mesh);
			Dictionary skin = fixture_skin(1, 1);
			skin["max_distance"] = 0.0;
			REQUIRE(server->soft_body_set_extra_property(fault, "skin/config", skin));
			server->soft_body_set_space(fault, space.rid());
			if (initialized) {
				space.step();
			}
			Array far;
			far.push_back(Transform3D(Basis(Vector3(2.5e38f, 2.5e38f, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)), Vector3()));
			REQUIRE(server->soft_body_set_extra_property(fault, "skin/pose", far));
			REQUIRE(Probe::set_body_frame(fault, Transform3D(Basis(Vector3(0, 0, 1), Math::PI / 4), Vector3(3, 4, 5))));
			ErrorCapture errors;
			space.step();
			CHECK(errors.has("SBREM-POSE"));
			REQUIRE_FALSE(Probe::in_space(fault));
			const auto before = Probe::build_counts(healthy.rid());
			server->free_rid(fault); // No recovery or detach before free.
			errors.clear();
			space.step(3);
			CHECK(errors.count() == 0);
			CHECK(Probe::build_counts(healthy.rid()).solver_steps == before.solver_steps + 3);
			CHECK(Probe::all_finite(healthy.rid()));
			const RID fresh = server->soft_body_create();
			for (const char *key : REMAINING_CONFIG_KEYS) {
				CHECK(server->soft_body_get_extra_property(fresh, key).get_type() == Variant::NIL);
			}
			CHECK(server->soft_body_get_extra_property(fresh, "skin/pose").get_type() == Variant::NIL);
			CHECK(Probe::build_counts(fresh).generation == 0);
			server->soft_body_set_mesh(fresh, mesh);
			REQUIRE(server->soft_body_set_extra_property(fresh, "skin/config", skin));
			server->soft_body_set_space(fresh, space.rid());
			space.step();
			CHECK(Probe::build_counts(fresh).skin_init == 1);
			CHECK(Probe::world_position(fresh, 0).distance_to(Vector3()) <= 1e-4f);
			server->free_rid(fresh);
			RenderingServer::get_singleton()->free_rid(mesh);
		}
	}

	TEST_CASE("remaining R09 preflight ordering") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		const auto fixture = fixture_seam();
		MeshBodyScope body(server.ptr(), space.rid(), &fixture);
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "vertex/config", remaining_config(3)));
		for (int kind = 0; kind < 3; ++kind) {
			CAPTURE(kind);
			Dictionary bad = remaining_config(3);
			if (kind == 0) {
				bad["misspelled"] = 1;
			}
			if (kind == 1) {
				bad["edge_scale"] = true;
			}
			if (kind == 2) {
				PackedFloat32Array values = make_floats(7, 1);
				values.set(4, 2);
				bad["edge_scale"] = values;
			}
			const auto before = Probe::build_counts(body.rid());
			RemainingSnapshot snapshot(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), "vertex/config", bad));
			CHECK(errors.count() == 1);
			snapshot.unchanged();
			const auto after = Probe::build_counts(body.rid());
			CHECK(after.body_creations == before.body_creations);
			CHECK(after.renderer_reads == before.renderer_reads);
			CHECK(after.create_constraints == before.create_constraints);
		}
	}

	TEST_CASE("remaining R10 typed neighbors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		Dictionary config;
		config["gravity_factor"] = 0.0;
		REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", config));
		server->soft_body_set_pressure_coefficient(body.rid(), 0.25);
		server->soft_body_set_damping_coefficient(body.rid(), 0.03);
		server->soft_body_set_simulation_precision(body.rid(), 30);
		server->soft_body_set_total_mass(body.rid(), 2.0);
		const auto actual = Probe::scalars(body.rid());
		CHECK(actual.pressure == doctest::Approx(0.25));
		CHECK(actual.damping == doctest::Approx(0.03));
		CHECK(actual.iterations == 30);
		CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/config")) == config);
		for (int i = 0; i < 4; ++i) {
			CHECK(Probe::vertex_inv_mass(body.rid(), i) == 2.0f);
		}
		const Vector3 start = Probe::world_position(body.rid(), 0);
		server->soft_body_apply_central_impulse(body.rid(), Vector3(2, 0, 0));
		space.step(30);
		CHECK(Probe::world_position(body.rid(), 0).x > start.x + 1e-3f);
		CHECK(Probe::all_finite(body.rid()));
		for (bool impulse : { false, true }) {
			CAPTURE(impulse);
			SpaceScope rotated_space(server.ptr());
			server->area_set_param(rotated_space.rid(), PS3DE::AREA_PARAM_GRAVITY, impulse ? 0.0 : 9.8);
			MeshBodyScope rotated(server.ptr(), rotated_space.rid());
			server->soft_body_set_state(rotated.rid(), PS3DE::BODY_STATE_TRANSFORM, Transform3D(Basis(Vector3(0, 0, 1), Math::PI / 2), Vector3(3, 4, 5)));
			REQUIRE(server->soft_body_set_extra_property(rotated.rid(), "bend/config", remaining_config(1)));
			const Vector3 initial = Probe::world_position(rotated.rid(), 0);
			if (impulse) {
				server->soft_body_apply_central_impulse(rotated.rid(), Vector3(1, 0, 0));
			}
			rotated_space.step(30);
			const Vector3 displacement = Probe::world_position(rotated.rid(), 0) - initial;
			if (impulse) {
				CHECK(displacement.x > 1e-3f);
				CHECK(Math::abs(displacement.y) <= 1e-4f);
			} else {
				CHECK(displacement.y < -1e-3f);
				CHECK(Math::abs(displacement.x) <= 1e-4f);
			}
		}
	}

	TEST_CASE("remaining R11 bounded allocation") {
		String error;
		for (uint64_t count : { uint64_t(65536), uint64_t(65537), UINT64_MAX }) {
			CAPTURE(count);
			CHECK(SoftBodyCapValidation::resource_size(count, 1, 65536, 4, &error, "skin/config.vertices") == (count == 65536));
			if (count != 65536) {
				CHECK(error.contains("SBREM-SIZE"));
			}
		}
		CHECK_FALSE(SoftBodyCapValidation::resource_size(uint64_t(INT32_MAX) / 4 + 1, 1, UINT64_MAX, 4, &error, "4*K"));
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		MeshBodyScope body(server.ptr(), space.rid());
		for (int cap : { 1, 2, 3, 4, 5 }) {
			CAPTURE(cap);
			Dictionary config = remaining_config(cap);
			if (cap == 1) {
				config["compliance"] = make_floats(65537, 1);
			}
			if (cap == 2) {
				config["max_distance_multiplier"] = make_floats(65537, 1);
			}
			if (cap == 3) {
				config["edge_scale"] = make_floats(65537, 1);
			}
			if (cap == 4) {
				PackedVector3Array vertices;
				vertices.resize(65537);
				config["vertices"] = vertices;
			}
			if (cap == 5) {
				Array matrices;
				matrices.resize(1025);
				config["inv_bind"] = matrices;
			}
			const auto before = Probe::build_counts(body.rid());
			RemainingSnapshot snapshot(server.ptr(), body.rid());
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
			CHECK(errors.has("SBREM-SIZE"));
			snapshot.unchanged();
			CHECK(Probe::build_counts(body.rid()).body_creations == before.body_creations);
			CHECK(Probe::build_counts(body.rid()).renderer_reads == before.renderer_reads);
		}
	}

	TEST_CASE("remaining R12 optional defaults") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		for (bool neighbor : { false, true }) {
			CAPTURE(neighbor);
			for (int cap = 0; cap < 6; ++cap) {
				CAPTURE(cap);
				MeshBodyScope body(server.ptr(), space.rid());
				Dictionary config = remaining_config(cap), reduced = config.duplicate(true);
				if (neighbor && cap != 0) {
					REQUIRE(server->soft_body_set_extra_property(body.rid(), "scalar/config", remaining_config(0)));
				} else if (neighbor) {
					REQUIRE(server->soft_body_set_extra_property(body.rid(), "bend/config", remaining_config(1)));
				}
				if (cap == 0) {
					config["gravity_factor"] = 2.0;
				}
				if (cap == 1) {
					config["type"] = 0;
				}
				if (cap == 2) {
					server->soft_body_pin_point(body.rid(), 0, true);
					config["max_distance_multiplier"] = 2.0;
				}
				if (cap == 3) {
					config["shear_scale"] = 5.0;
					reduced.erase("shear_scale");
				}
				if (cap == 4) {
					config["compliance"] = 0.2;
					PackedInt32Array fixed;
					fixed.push_back(0);
					config["fixed"] = fixed;
				}
				if (cap == 5) {
					config["max_distance"] = 0.1;
					config["back_stop_distance"] = 0.0;
					config["back_stop_radius"] = 2.0;
				}
				REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], config));
				REQUIRE(server->soft_body_set_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap], reduced));
				CHECK(SoftBodyCapValidation::equal(server->soft_body_get_extra_property(body.rid(), REMAINING_CONFIG_KEYS[cap]), reduced));
				if (cap == 0) {
					CHECK(Probe::scalars(body.rid()).gravity_factor == 1.0f);
				}
				if (cap == 1) {
					CHECK(Probe::dihedral_bend_count(body.rid()) == 1);
				}
				if (cap == 2) {
					for (int i = 0; i < 3; ++i) {
						const auto lra = Probe::lra(body.rid(), i);
						CHECK(lra.rest == doctest::Approx(Probe::vertex_position(body.rid(), lra.vertices[1]).length()).epsilon(1e-5));
					}
				}
				if (cap == 3) {
					const double base = (1.0 / 60 / 20) * (1.0 / 60 / 20) * 8;
					for (int i = 0; i < 6; ++i) {
						CHECK(Probe::edge(body.rid(), i).compliance == doctest::Approx(base).epsilon(1e-5));
					}
				}
				if (cap == 4) {
					CHECK(Probe::tetra(body.rid(), 0).compliance == 0);
					CHECK(Probe::vertex_inv_mass(body.rid(), 0) == 4);
				}
				if (cap == 5) {
					for (int i = 0; i < 4; ++i) {
						const auto skin = Probe::skin(body.rid(), i);
						CHECK(skin.max_distance == doctest::Approx(0.05));
						CHECK(skin.back_stop_distance == FLT_MAX);
						CHECK(skin.back_stop_radius == 40);
					}
				}
				if (neighbor && cap != 0) {
					CHECK(Dictionary(server->soft_body_get_extra_property(body.rid(), "scalar/config")) == remaining_config(0));
					CHECK(Probe::scalars(body.rid()).friction == doctest::Approx(0.4));
					CHECK(Probe::scalars(body.rid()).restitution == doctest::Approx(0.2));
				} else if (neighbor) {
					CHECK(Probe::dihedral_bend_count(body.rid()) == 1);
				}
			}
		}
	}

} // TEST_SUITE
} // namespace TestJoltSoftBodyCap
