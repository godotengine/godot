/**************************************************************************/
/*  test_soft_body_capabilities.h                                         */
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

#include "soft_body_cap_test_support.h"

#include <cmath>

namespace TestJoltSoftBodyCap {

TEST_SUITE("[JoltSoftBodyCap]") {
	/* ---------------------------------------------------------------- protocol */

	TEST_CASE("unknown prefix is silent") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, "nope/x", 1));
		CHECK(errors.count() == 0);

		server->free_rid(body);
	}

	TEST_CASE("misspelled key errors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, "rod/joint", make_joints(9)));
		CHECK(errors.has("rod/joint"));
		CHECK(errors.has("not a valid key"));

		server->free_rid(body);
	}

	TEST_CASE("wrong type errors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", 5)));
		CHECK(errors.has("PackedVector3Array"));
		CHECK(errors.has("int"));
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, 5));
		CHECK(errors.has("Dictionary"));
		CHECK(errors.has("SBREM-TYPE"));

		server->free_rid(body);
	}

	TEST_CASE("get of unknown prefix is empty") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		CHECK(server->soft_body_get_extra_property(body, "nope/x").get_type() == Variant::NIL);

		server->free_rid(body);
	}

	TEST_CASE("jolt lists rod properties") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		const TypedArray<Dictionary> list = server->soft_body_get_extra_property_list(body);
		CHECK(list.size() == 19);
		for (const char *key : { K_CONFIG, K_STATE }) {
			CHECK_MESSAGE(!find_property(list, key).is_empty(), key);
		}
		for (const char *key : { "rod/joints", "rod/compliance", "rod/bend", "rod/fixed" }) {
			CHECK(find_property(list, key).is_empty());
		}

		server->free_rid(body);
	}

	TEST_CASE("godot physics lists nothing") {
		ServerScope server("GodotPhysics3D");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		CHECK(server->soft_body_get_extra_property_list(body).size() == 0);

		server->free_rid(body);
	}

	TEST_CASE("rod state is not serialised") { // codespell:ignore serialised
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		const Dictionary entry = find_property(server->soft_body_get_extra_property_list(body), K_STATE);
		REQUIRE_FALSE(entry.is_empty());
		CHECK(((uint32_t)entry["usage"] & PROPERTY_USAGE_STORAGE) == 0);

		server->free_rid(body);
	}

	TEST_CASE("rod config is serialised") { // codespell:ignore serialised
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		const Dictionary entry = find_property(server->soft_body_get_extra_property_list(body), K_CONFIG);
		REQUIRE_FALSE(entry.is_empty());
		CHECK(((uint32_t)entry["usage"] & PROPERTY_USAGE_STORAGE) != 0);

		server->free_rid(body);
	}

	TEST_CASE("wrapper forwards all three") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());

		// All three calls must pass through the wrapper, not the backend directly.
		CHECK_FALSE(Probe::is_inner_jolt_server(server.ptr()));

		RID body = server->soft_body_create();
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);
		CHECK(server->soft_body_get_extra_property_list(body).size() >= 5);

		server->free_rid(body);
	}

	TEST_CASE("threaded wrapper forwards") {
		ProjectSettings *settings = ProjectSettings::get_singleton();
		const String key = "physics/3d/run_on_separate_thread";
		const Variant saved = settings->has_setting(key) ? settings->get_setting(key) : Variant(false);
		settings->set_setting(key, true);

		{
			// The flag is read when the server is constructed, not when it steps.
			ServerScope server("Jolt Physics");
			REQUIRE(server.is_valid());

			RID body = server->soft_body_create();
			CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
			CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);
			CHECK(server->soft_body_get_extra_property_list(body).size() >= 5);
			server->free_rid(body);
		}

		settings->set_setting(key, saved);
	}

	TEST_CASE("writable keys read back") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		const PackedVector3Array joints = make_joints(9);
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)));

		const PackedVector3Array read = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant());
		REQUIRE(read.size() == joints.size());
		for (int i = 0; i < joints.size(); i++) {
			CHECK(read[i] == joints[i]);
		}

		server->free_rid(body);
	}

	TEST_CASE("rod state cannot be written") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_STATE, make_floats(32)));
		CHECK(errors.has("read-only"));

		server->free_rid(body);
	}

	TEST_CASE("invalid rid is rejected") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());

		ErrorCapture errors;
		SUBCASE("set") {
			CHECK_FALSE(server->soft_body_set_extra_property(RID(), K_CONFIG, rod_candidate(server.ptr(), RID(), "joints", make_joints(9))));
		}
		SUBCASE("get") {
			CHECK(Dictionary(server->soft_body_get_extra_property(RID(), K_CONFIG)).get("joints", Variant()).get_type() == Variant::NIL);
		}
		SUBCASE("list") {
			CHECK(server->soft_body_get_extra_property_list(RID()).size() == 0);
		}
		CHECK(errors.count() > 0);
	}

	TEST_CASE("freed rid is rejected") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->free_rid(body);

		ErrorCapture errors;
		SUBCASE("set") {
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		}
		SUBCASE("get") {
			CHECK(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant()).get_type() == Variant::NIL);
		}
		SUBCASE("list") {
			CHECK(server->soft_body_get_extra_property_list(body).size() == 0);
		}
		CHECK(errors.count() > 0);
	}

	TEST_CASE("rod state needs a space") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));

		ErrorCapture errors;
		CHECK(server->soft_body_get_extra_property(body, K_STATE).get_type() == Variant::NIL);
		CHECK(errors.count() > 0);

		server->free_rid(body);
	}

	TEST_CASE("writable keys read without a space") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));

		ErrorCapture errors;
		// Stored keys must serialize before the body enters a space.
		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);
		CHECK(errors.count() == 0);

		server->free_rid(body);
	}

	TEST_CASE("rod state is valid before stepping") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		const PackedFloat32Array state = server->soft_body_get_extra_property(body, K_STATE);
		REQUIRE(state.size() == 8 * 4);
		for (int i = 0; i < state.size(); i++) {
			CHECK(Math::is_finite(state[i]));
		}

		server->free_rid(body);
	}

	TEST_CASE("godot physics ignores rod keys") {
		ServerScope server("GodotPhysics3D");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		CHECK(errors.count() == errors.warning_count());
		CHECK(errors.warning_count() <= 2);
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		CHECK(server->soft_body_get_extra_property_list(body).is_empty());
		CHECK(errors.count() == errors.warning_count());
		CHECK(errors.warning_count() <= 1);
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		CHECK(server->soft_body_get_extra_property_list(body).is_empty());
		CHECK(errors.count() == 0);

		server->free_rid(body);
	}

	TEST_CASE("extension shims forward") {
		// Without GDExtension overrides, each EXBIND*R call reports one error and
		// returns its type's default. This checks all three bindings in every build.
		PhysicsServer3D *saved = PhysicsServer3D::get_singleton();
		PhysicsServer3DExtension *server = memnew(PhysicsServer3DExtension);
		const RID body;
		{
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_config(9)));
			CHECK(server->soft_body_get_extra_property(body, K_CONFIG).get_type() == Variant::NIL);
			CHECK(server->soft_body_get_extra_property_list(body).is_empty());
			CHECK(errors.count() == 3);
		}
		memdelete(server);
		PhysicsServer3D::set_singleton_for_tests(saved);

		// ClassDB records virtual signatures only under DEBUG_ENABLED.
#ifdef DEBUG_ENABLED
		List<MethodInfo> methods;
		ClassDB::get_virtual_methods("PhysicsServer3DExtension", &methods);

		HashMap<String, int> arg_counts;
		for (const MethodInfo &method : methods) {
			arg_counts[method.name] = method.arguments.size();
		}

		REQUIRE(arg_counts.has("_soft_body_set_extra_property"));
		CHECK(arg_counts["_soft_body_set_extra_property"] == 3);
		REQUIRE(arg_counts.has("_soft_body_get_extra_property"));
		CHECK(arg_counts["_soft_body_get_extra_property"] == 2);
		REQUIRE(arg_counts.has("_soft_body_get_extra_property_list"));
		CHECK(arg_counts["_soft_body_get_extra_property_list"] == 1);
#endif
	}

	TEST_CASE("dummy server answers emptily") {
		PhysicsServer3D *saved = PhysicsServer3D::get_singleton();
		PhysicsServer3DDummy *server = memnew(PhysicsServer3DDummy);

		RID body = server->soft_body_create();
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server, body, "joints", make_joints(9))));
		CHECK(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant()).get_type() == Variant::NIL);
		CHECK(server->soft_body_get_extra_property_list(body).size() == 0);

		memdelete(server);
		PhysicsServer3D::set_singleton_for_tests(saved);
	}

	/* -------------------------------------------------------------- validation */

	TEST_CASE("one joint is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(1))));
		CHECK(errors.has("SBREM-SIZE"));
		CHECK(errors.has("joint count 1 requires 2..1024"));

		server->free_rid(body);
	}

	TEST_CASE("compliance count mismatch") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(7))));
		CHECK_MESSAGE(errors.has("has 7 entries"), errors.joined());
		CHECK_MESSAGE(errors.has("has 8 segments"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("bend count mismatch") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(8))));

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "bend", make_floats(8))));
		CHECK_MESSAGE(errors.has("has 8 entries"), errors.joined());
		CHECK_MESSAGE(errors.has("has 7 bend-twist"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("negative fixed is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "fixed", -1)));
		CHECK_MESSAGE(errors.has("[0, 9]"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("over-range fixed is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "fixed", 10)));
		CHECK_MESSAGE(errors.has("[0, 9]"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("zero fixed is a free rod") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9, 0));
		server->soft_body_set_space(body, space.rid());
		// Read after a step, so `_update_mass()` has already rewritten every
		// inverse mass at least once.
		space.step();
		REQUIRE(Probe::in_space(body));

		for (int i = 0; i < 9; i++) {
			CHECK(Probe::vertex_inv_mass(body, i) > 0.0f);
		}

		server->free_rid(body);
	}

	TEST_CASE("fully pinned rod") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9, 9));
		server->soft_body_set_space(body, space.rid());
		space.step();
		REQUIRE(Probe::in_space(body));

		for (int i = 0; i < 9; i++) {
			CHECK(Probe::vertex_inv_mass(body, i) == 0.0f);
		}

		server->free_rid(body);
	}

	TEST_CASE("two joint rod needs no bend") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(2))));
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(1))));
		// One segment carries zero bend-twist constraints, not -1 of them.
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "bend", make_floats(0))));

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "bend", make_floats(1))));

		server->free_rid(body);
	}

	TEST_CASE("valid rod is accepted") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(8))));
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "bend", make_floats(7))));
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "fixed", 1)));

		server->free_rid(body);
	}

	TEST_CASE("zero length segment is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;

		SUBCASE("coincident") {
			PackedVector3Array joints = make_joints(3);
			joints.set(1, joints[0]);
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)));
			CHECK_MESSAGE(errors.has("Segment 0"), errors.joined());
		}
		SUBCASE("below the epsilon") {
			PackedVector3Array joints = make_joints(3);
			joints.set(1, joints[0] + Vector3(5e-7, 0, 0));
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)));
			CHECK_MESSAGE(errors.has("Segment 0"), errors.joined());
		}
		SUBCASE("above the epsilon") {
			PackedVector3Array joints = make_joints(3);
			joints.set(1, joints[0] + Vector3(2e-6, 0, 0));
			CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)));
			CHECK(errors.count() == 0);
		}
		SUBCASE("adjacent representable values around minimum") {
			for (real_t endpoint : { std::nextafter(real_t(1e-6), real_t(0)), std::nextafter(real_t(1e-6), real_t(1)) }) {
				PackedVector3Array joints = make_joints(2, endpoint);
				const bool valid = double(joints[1].length_squared()) >= 1e-12;
				CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)) == valid);
			}
		}
		SUBCASE("finite coordinates cannot overflow exported float length") {
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(2, real_t(1e20)))));
			CHECK(errors.has("SBREM-VALUE"));
			CHECK(errors.has("squared length"));
		}

		server->free_rid(body);
	}

	TEST_CASE("non finite joints are refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		real_t value = 0.0;
		SUBCASE("nan") {
			value = NAN;
		}
		SUBCASE("inf") {
			value = INFINITY;
		}
		SUBCASE("overflowing literal") {
			// Keep overflow at runtime.
			volatile double large = 1e200;
			value = (real_t)(large * large);
		}

		PackedVector3Array joints = make_joints(4);
		joints.set(2, Vector3(value, 0, 0));
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", joints)));
		CHECK_MESSAGE(errors.has("entry 2"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("negative compliance is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));

		PackedFloat32Array compliance = make_floats(8);
		compliance.set(3, -1.0f);

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", compliance)));
		CHECK_MESSAGE(errors.has("entry 3"), errors.joined());
		CHECK_MESSAGE(errors.has("negative"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("non finite compliance is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(8))));

		ErrorCapture errors;
		SUBCASE("compliance") {
			PackedFloat32Array values = make_floats(8);
			values.set(2, NAN);
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", values)));
		}
		SUBCASE("bend") {
			PackedFloat32Array values = make_floats(7);
			values.set(2, NAN);
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "bend", values)));
		}
		CHECK_MESSAGE(errors.has("entry 2"), errors.joined());
		CHECK_MESSAGE(errors.has("finite"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("fixed without required joints is rejected") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "fixed", 5)));
		CHECK(errors.has("SBREM-KEY"));
		CHECK(errors.has("joints"));
		CHECK(server->soft_body_get_extra_property(body, K_CONFIG).get_type() == Variant::NIL);
		CHECK(set_valid_rod(server.ptr(), body, 9, 5));
		CHECK((int)Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("fixed", Variant()) == 5);
		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);

		server->free_rid(body);
	}

	TEST_CASE("rod topology replacement is atomic") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		space.step(3);
		RemainingSnapshot before(server.ptr(), body);
		const auto generation = Probe::build_counts(body).generation;

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(3))));
		CHECK_MESSAGE(errors.has("has 8 entries"), errors.joined());
		CHECK_MESSAGE(errors.has("has 2 segments"), errors.joined());
		before.unchanged();
		// A complete consistent topology replacement is atomic and resets live data.
		REQUIRE(set_valid_rod(server.ptr(), body, 3));
		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG))["joints"]).size() == 3);
		CHECK(Probe::vertex_count(body) == 3);
		CHECK(Probe::rod_count(body) == 2);
		CHECK(Probe::build_counts(body).generation == generation + 1);
		CHECK(PackedFloat32Array(server->soft_body_get_extra_property(body, K_STATE)).size() == 8);
		CHECK(Probe::vertex_position(body, 2) == Vector3(2, 0, 0));
		CHECK(Probe::velocity(body, 2) == Vector3());

		server->free_rid(body);
	}

	TEST_CASE("rejected set leaves state untouched") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));

		ErrorCapture errors;
		REQUIRE_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(3))));

		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);
		CHECK(PackedFloat32Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("compliance", Variant())).size() == 8);
		CHECK(PackedFloat32Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("bend", Variant())).size() == 7);
		CHECK((int)Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("fixed", Variant()) == 1);
		CHECK(server->soft_body_get_extra_property(body, K_STATE).get_type() == Variant::NIL);

		server->free_rid(body);
	}

	TEST_CASE("re-setting a key replaces it") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9))));
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(4))));

		const PackedVector3Array read = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant());
		CHECK(read.size() == 4);

		server->free_rid(body);
	}

	TEST_CASE("over-long rod is refused") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(1025))));
		CHECK_MESSAGE(errors.has("1024"), errors.joined());

		server->free_rid(body);
	}

	TEST_CASE("a rod at the cap is accepted") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(1024))));
		CHECK(errors.count() == 0);

		server->free_rid(body);
	}

	TEST_CASE("two joint rod orientation is bounded") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 2));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		// A single segment has no adjacent rod, so it carries no bend-twist
		// constraint at all.
		CHECK(Probe::rod_count(body) == 1);
		CHECK(Probe::rod_bend_twist_count(body) == 0);

		server->soft_body_apply_central_impulse(body, Vector3(0, -20, 0));
		space.step(60);

		const PackedFloat32Array state = server->soft_body_get_extra_property(body, K_STATE);
		REQUIRE(state.size() == 4);
		for (int i = 0; i < state.size(); i++) {
			CHECK(Math::is_finite(state[i]));
		}

		server->free_rid(body);
	}

	TEST_CASE("fixed without joints does not activate the rod") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID mesh = make_quad_mesh();

		RID body = server->soft_body_create();
		{
			ErrorCapture errors;
			CHECK_FALSE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "fixed", 1)));
			CHECK(errors.has("SBREM-KEY"));
		}
		server->soft_body_set_mesh(body, mesh);
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		CHECK(Probe::rod_count(body) == 0);
		CHECK(Probe::face_count(body) == 2);

		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		CHECK(Probe::rod_count(body) == 8);
		CHECK((int)Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("fixed", Variant()) == 1);

		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	/* ------------------------------------------------------------- arbitration */

	TEST_CASE("two replacing capabilities conflict") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		RemainingSnapshot before(server.ptr(), body);
		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, "test_replace/enabled", true));
		CHECK_MESSAGE(errors.has("SBREM-CONFLICT"), errors.joined());
		before.unchanged();
		CHECK(Probe::in_space(body));

		server->free_rid(body);
	}

	TEST_CASE("cloth without a mesh still errors") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();

		ErrorCapture errors;
		server->soft_body_set_space(body, space.rid());
		// Stock behavior: cloth without a mesh is inert and emits no diagnostic.
		CHECK(errors.count() == 0);
		CHECK_FALSE(Probe::in_space(body));

		server->free_rid(body);
	}

	TEST_CASE("cloth regression") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID mesh = make_quad_mesh();

		RID body = server->soft_body_create();
		server->soft_body_set_mesh(body, mesh);
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		CHECK(Probe::vertex_count(body) == 4);
		CHECK(Probe::face_count(body) == 2);
		CHECK(Probe::edge_count(body) > 0);
		CHECK(Probe::rod_count(body) == 0);

		const Vector3 before = Probe::vertex_position(body, 0);
		server->soft_body_apply_central_impulse(body, Vector3(0, -20, 0));
		space.step(30);
		CHECK(Probe::vertex_position(body, 0) != before);

		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	// Without capabilities, EBendType::None and FLT_MAX disable the quad's bend
	// constraint. Other capability arrays also stay empty.
	TEST_CASE("a cloth has edges but cannot bend") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID mesh = make_quad_mesh();

		RID body = server->soft_body_create();
		server->soft_body_set_mesh(body, mesh);

		SUBCASE("default cloth") {}
		SUBCASE("stiffest cloth") {
			server->soft_body_set_linear_stiffness(body, 1.0);
			server->soft_body_set_simulation_precision(body, 20);
		}
		SUBCASE("every knob at once") {
			server->soft_body_set_linear_stiffness(body, 1.0);
			server->soft_body_set_shrinking_factor(body, 0.5);
			server->soft_body_set_pressure_coefficient(body, 100.0);
			server->soft_body_set_damping_coefficient(body, 1.0);
			server->soft_body_set_drag_coefficient(body, 1.0);
			server->soft_body_set_simulation_precision(body, 20);
		}

		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		// Confirm the body was built before checking absent constraints.
		CHECK(Probe::edge_count(body) > 0);
		CHECK(Probe::face_count(body) == 2);

		CHECK(Probe::dihedral_bend_count(body) == 0);
		CHECK(Probe::lra_count(body) == 0);
		CHECK(Probe::volume_count(body) == 0);
		CHECK(Probe::skinned_count(body) == 0);

		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	// Prototype flat keys are not aliases for atomic configuration dictionaries.
	TEST_CASE("prototype flat keys do not configure constraint arrays") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		RID body = server->soft_body_create();

		ErrorCapture errors;
		CHECK_FALSE(server->soft_body_set_extra_property(body, "bend/compliance", 1e-4));
		CHECK(errors.has("SBREM-KEY"));
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, "lra/type", 1));
		CHECK(errors.has("SBREM-KEY"));
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, "volume/tetrahedra", PackedInt32Array()));
		CHECK(errors.has("SBREM-KEY"));
		errors.clear();
		CHECK_FALSE(server->soft_body_set_extra_property(body, "skin/joint_weights", PackedFloat32Array()));
		CHECK(errors.has("SBREM-KEY"));

		CHECK(server->soft_body_get_extra_property(body, "bend/compliance").get_type() == Variant::NIL);

		server->free_rid(body);
	}

	TEST_CASE("mesh with a rod warns") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID mesh = make_quad_mesh();

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_mesh(body, mesh);

		ErrorCapture errors;
		server->soft_body_set_space(body, space.rid());
		CHECK_MESSAGE(errors.has("mesh is ignored"), errors.joined());
		REQUIRE(Probe::in_space(body));
		CHECK(Probe::rod_count(body) == 8);
		CHECK(Probe::face_count(body) == 0);
		CHECK(Probe::vertex_count(body) == 9);

		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	/* ------------------------------------------------------------------- traps */

	TEST_CASE("optimize makes the rod simulate") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		const Vector3 before = Probe::vertex_position(body, 8);
		server->soft_body_apply_central_impulse(body, Vector3(0, -20, 0));
		space.step(30);
		CHECK(Probe::vertex_position(body, 8).distance_to(before) > 0.01);

		// Missing Optimize() leaves no solver update groups, allowing unbounded stretch.
		const real_t span = Probe::vertex_position(body, 0).distance_to(Probe::vertex_position(body, 8));
		CHECK(span > 7.0);
		CHECK(span < 9.0);

		server->free_rid(body);
	}

	TEST_CASE("rod without update position drifts") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		// A rod is anchored to the static world, so the body frame must not chase
		// its own vertices.
		CHECK_FALSE(Probe::update_position(body));

		const Vector3 origin = Probe::center_of_mass(body);
		server->soft_body_apply_central_impulse(body, Vector3(0, -20, 0));
		space.step(60);

		CHECK(Probe::center_of_mass(body).distance_to(origin) < 0.001);
		// The rest length is untouched by the frame decision.
		CHECK(Math::is_equal_approx(Probe::rod_length(body, 0), 1.0f));

		server->free_rid(body);
	}

	TEST_CASE("update position set too early drifts on rebuild") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		space.step();

		// _add_to_space must restore the flag each rebuild; settings do not survive.
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(8, 0.5f))));
		REQUIRE(Probe::in_space(body));
		CHECK_FALSE(Probe::update_position(body));

		const Vector3 origin = Probe::center_of_mass(body);
		server->soft_body_apply_central_impulse(body, Vector3(0, -20, 0));
		space.step(60);
		CHECK(Probe::center_of_mass(body).distance_to(origin) < 0.001);

		server->free_rid(body);
	}

	TEST_CASE("mass update preserves pins") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9, 2));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		// Restore pins after _update_mass() overwrites inverse masses.
		server->soft_body_set_total_mass(body, 5.0);
		space.step();

		CHECK(Probe::vertex_inv_mass(body, 0) == 0.0f);
		CHECK(Probe::vertex_inv_mass(body, 1) == 0.0f);
		CHECK(Probe::vertex_inv_mass(body, 2) > 0.0f);

		server->free_rid(body);
	}

	TEST_CASE("calculate rod properties keeps chain order") {
		CHECK(Probe::calculate_rod_properties_keeps_chain_order(9));
	}

	TEST_CASE("optimize is identity for a nine vertex chain") {
		CHECK(Probe::optimize_is_identity_for_chain(9));
	}

	TEST_CASE("inverse mass matches update mass") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9, 0));
		server->soft_body_set_total_mass(body, 1.0);
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		space.step();

		// The cloth path's formula, not 1.0: vertex count over total mass.
		CHECK(Math::is_equal_approx(Probe::vertex_inv_mass(body, 4), 9.0f));

		server->free_rid(body);
	}

	TEST_CASE("rebuild rederives rod properties") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		CHECK(Math::is_equal_approx(Probe::rod_length(body, 0), 1.0f));

		// A reused settings object would short-circuit `CalculateRodProperties`,
		// leaving the previous build's lengths behind.
		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "joints", make_joints(9, 2.0))));
		REQUIRE(Probe::in_space(body));
		CHECK(Math::is_equal_approx(Probe::rod_length(body, 0), 2.0f));
		CHECK(Probe::all_bishop_frames_set(body));

		server->free_rid(body);
	}

	TEST_CASE("state order survives optimize at scale") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		const int joint_count = 200;
		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, joint_count));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		REQUIRE(Probe::rod_count(body) == joint_count - 1);

		// Every +X segment's Bishop frame must map +Z to +X in rod/state.
		for (int i = 0; i < joint_count - 1; i++) {
			CHECK(Probe::rod_vertex(body, i, 0) == i);
			CHECK(Probe::rod_vertex(body, i, 1) == i + 1);

			const Quaternion rotation = read_rod_rotation(server.ptr(), body, i);
			const Vector3 tangent = rotation.xform(Vector3(0, 0, 1));
			CHECK(tangent.distance_to(Vector3(1, 0, 0)) < 0.01);
		}

		server->free_rid(body);
	}

	/* --------------------------------------------------------------- derivation */

	TEST_CASE("derive populates rod properties") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		CHECK(Probe::min_rod_length(body) > 0.0f);
		CHECK(Probe::all_bishop_frames_set(body));

		server->free_rid(body);
	}

	TEST_CASE("rod states are sized by segment") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		// Segments, not particles: 9 joints give 8 rods.
		CHECK(Probe::vertex_count(body) == 9);
		CHECK(Probe::rod_count(body) == 8);
		CHECK(Probe::rod_state_is_readable(body, 7));
		CHECK(PackedFloat32Array(server->soft_body_get_extra_property(body, K_STATE)).size() == 8 * 4);

		server->free_rid(body);
	}

	TEST_CASE("one pin frees orientation two pins fix it") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID free_end = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), free_end, 9, 1));
		server->soft_body_set_space(free_end, space.rid());
		REQUIRE(Probe::in_space(free_end));

		RID clamped = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), clamped, 9, 2));
		server->soft_body_set_space(clamped, space.rid());
		REQUIRE(Probe::in_space(clamped));

		const Quaternion clamped_before = read_rod_rotation(server.ptr(), clamped, 0);

		server->soft_body_apply_central_impulse(free_end, Vector3(0, -40, 0));
		server->soft_body_apply_central_impulse(clamped, Vector3(0, -40, 0));
		space.step(60);

		// One fixed joint permits motion; two clamp rod 0. Compare positions:
		// measured swing is 0.014 units, while clamped values stay bit-identical.
		CHECK(Probe::vertex_position(free_end, 1) != Vector3(1, 0, 0));
		CHECK(Probe::vertex_position(clamped, 1) == Vector3(1, 0, 0));

		// Rotation stays bit-identical in both bodies at float precision; use it
		// only for the clamped check. Point impulses require the rod's absent mesh map.
		CHECK(read_rod_rotation(server.ptr(), clamped, 0) == clamped_before);

		server->free_rid(clamped);
		server->free_rid(free_end);
	}

	/* --------------------------------------------------------------- lifecycle */

	TEST_CASE("set mesh keeps capability state") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());
		RID mesh = make_quad_mesh();

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));

		ErrorCapture errors;
		server->soft_body_set_mesh(body, mesh);
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);
		CHECK(Probe::rod_count(body) == 8);

		server->free_rid(body);
		RenderingServer::get_singleton()->free_rid(mesh);
	}

	TEST_CASE("space change keeps capability state") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		server->soft_body_set_space(body, RID());
		CHECK_FALSE(Probe::in_space(body));

		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		CHECK(Probe::rod_count(body) == 8);
		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())).size() == 9);

		server->free_rid(body);
	}

	TEST_CASE("new body starts with no capability state") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID first = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), first, 9));
		server->soft_body_set_space(first, space.rid());
		REQUIRE(Probe::in_space(first));
		server->free_rid(first);

		RID second = server->soft_body_create();
		CHECK(server->soft_body_get_extra_property(second, K_CONFIG).get_type() == Variant::NIL);
		CHECK(server->soft_body_get_extra_property_list(second).size() >= 5);

		server->free_rid(second);
	}

	TEST_CASE("live parameter change rebuilds") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));
		CHECK(Probe::rod_compliance(body, 0) == 0.0f);

		REQUIRE(server->soft_body_set_extra_property(body, K_CONFIG, rod_candidate(server.ptr(), body, "compliance", make_floats(8, 0.25f))));
		space.step();
		REQUIRE(Probe::in_space(body));
		CHECK(Math::is_equal_approx(Probe::rod_compliance(body, 0), 0.25f));

		server->free_rid(body);
	}

	TEST_CASE("cloth vertex api rejects a rod") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9, 2));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		ErrorCapture errors;
		// A rod has no mesh, so the mesh-indexed cloth API has nothing to address.
		server->soft_body_move_point(body, 0, Vector3(1, 1, 1));
		CHECK(server->soft_body_get_point_global_position(body, 0) == Vector3());
		server->soft_body_pin_point(body, 0, true);
		CHECK_FALSE(server->soft_body_is_point_pinned(body, 0));
		server->soft_body_apply_point_impulse(body, 0, Vector3(0, 1, 0));
		CHECK(errors.count() > 0);

		space.step();
		CHECK(Probe::vertex_inv_mass(body, 0) == 0.0f);
		CHECK(Probe::vertex_inv_mass(body, 1) == 0.0f);
		CHECK(Probe::vertex_inv_mass(body, 2) > 0.0f);

		server->free_rid(body);
	}

	TEST_CASE("cloth setters do not disturb a rod") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());
		SpaceScope space(server.ptr());

		RID body = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), body, 9));
		server->soft_body_set_space(body, space.rid());
		REQUIRE(Probe::in_space(body));

		const PackedVector3Array joints_before = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant());
		const PackedFloat32Array compliance_before = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("compliance", Variant());
		const PackedFloat32Array bend_before = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("bend", Variant());
		const int fixed_before = Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("fixed", Variant());
		const PackedFloat32Array state_before = server->soft_body_get_extra_property(body, K_STATE);

		const auto generation = Probe::build_counts(body).generation;
		// Valid typed writes preserve author config and rebuild derived live data.
		server->soft_body_set_simulation_precision(body, 7);
		server->soft_body_set_total_mass(body, 3.0);
		server->soft_body_set_linear_stiffness(body, 0.25);
		server->soft_body_set_shrinking_factor(body, 0.5);
		server->soft_body_set_pressure_coefficient(body, 2.0);
		server->soft_body_set_damping_coefficient(body, 0.75);

		CHECK(server->soft_body_get_simulation_precision(body) == 7);
		CHECK(Math::is_equal_approx(server->soft_body_get_total_mass(body), (real_t)3.0));
		CHECK(Math::is_equal_approx(server->soft_body_get_linear_stiffness(body), (real_t)0.25));
		CHECK(Math::is_equal_approx(server->soft_body_get_shrinking_factor(body), (real_t)0.5));
		CHECK(Math::is_equal_approx(server->soft_body_get_pressure_coefficient(body), (real_t)2.0));
		CHECK(Math::is_equal_approx(server->soft_body_get_damping_coefficient(body), (real_t)0.75));

		CHECK(PackedVector3Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("joints", Variant())) == joints_before);
		CHECK(PackedFloat32Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("compliance", Variant())) == compliance_before);
		CHECK(PackedFloat32Array(Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("bend", Variant())) == bend_before);
		CHECK((int)Dictionary(server->soft_body_get_extra_property(body, K_CONFIG)).get("fixed", Variant()) == fixed_before);
		CHECK(PackedFloat32Array(server->soft_body_get_extra_property(body, K_STATE)) == state_before);
		CHECK(Probe::build_counts(body).generation > generation);
		CHECK_FALSE(Probe::update_position(body));

		server->free_rid(body);
	}

	TEST_CASE("free before build leaks nothing") {
		ServerScope server("Jolt Physics");
		REQUIRE(server.is_valid());

		RID first = server->soft_body_create();
		REQUIRE(set_valid_rod(server.ptr(), first, 9));
		// Freed without ever entering a space, so nothing was ever built.
		server->free_rid(first);

		RID second = server->soft_body_create();
		CHECK(server->soft_body_get_extra_property(second, K_CONFIG).get_type() == Variant::NIL);

		server->free_rid(second);
	}

	/* ------------------------------------------------------------------ registry */

	TEST_CASE("every capability file is listed") {
		const int file_count = Probe::capability_source_file_count();
		REQUIRE(file_count > 0);

		HashSet<String> listed;
		for (int i = 0; i < Probe::capability_count(); i++) {
			listed.insert(Probe::capability_prefix(i));
		}

		// Every cap_<x>.cpp must register its <x>/ prefix.
		for (int i = 0; i < file_count; i++) {
			const String file = Probe::capability_source_file(i);
			REQUIRE(file.begins_with("cap_"));
			REQUIRE(file.ends_with(".cpp"));
			const String prefix = file.substr(4, file.length() - 8) + "/";
			CHECK_MESSAGE(listed.has(prefix), file);
		}
		CHECK(Probe::capability_count() == file_count);
	}

	TEST_CASE("every capability names a writable required key") {
		REQUIRE(Probe::capability_count() > 0);

		for (int cap = 0; cap < Probe::capability_count(); cap++) {
			const String required = Probe::capability_required_key(cap);
			CHECK_MESSAGE(required.begins_with(Probe::capability_prefix(cap)), required);

			bool found = false;
			for (int prop = 0; prop < Probe::capability_property_count(cap); prop++) {
				if (Probe::capability_property_name(cap, prop) == required) {
					found = true;
					// Activation requires a writable key.
					CHECK_FALSE(Probe::capability_property_live_read(cap, prop));
				}
			}
			CHECK_MESSAGE(found, required);
		}
	}

} // TEST_SUITE

} // namespace TestJoltSoftBodyCap
