/**************************************************************************/
/*  test_gdscript_coverage.h                                              */
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

#ifdef TOOLS_ENABLED

#include "../gdscript.h"

#include "tests/test_macros.h"

namespace GDScriptTests {

// Helper: reset coverage state to a clean baseline between test cases.
// Saves and restores config so tests do not interfere with each other.
struct CoverageScopedReset {
	GDScriptLanguage *lang = nullptr;
	GDScriptLanguage::CoverageMode saved_mode;
	GDScriptLanguage::CoverageFormat saved_format;
	String saved_output_path;
	float saved_threshold = 0.0f;
	Vector<String> saved_include;
	Vector<String> saved_exclude;
	bool saved_enabled = false;
	bool saved_written = false;

	CoverageScopedReset() {
		lang = GDScriptLanguage::get_singleton();
		if (!lang) {
			return;
		}
		saved_mode = lang->coverage_mode;
		saved_format = lang->coverage_format;
		saved_output_path = lang->coverage_output_path;
		saved_threshold = lang->coverage_threshold;
		saved_include = lang->coverage_include;
		saved_exclude = lang->coverage_exclude;
		saved_enabled = lang->coverage_enabled;
		saved_written = lang->coverage_written;

		// Start fresh so recorded data is isolated to this test.
		lang->coverage_hits.clear();
		lang->coverage_func_hits.clear();
		lang->coverage_branch_hits.clear();
		lang->coverage_include.clear();
		lang->coverage_exclude.clear();
		lang->coverage_enabled = true;
		lang->coverage_written = false;
		lang->coverage_threshold = 0.0f;
		lang->coverage_mode = GDScriptLanguage::COVERAGE_MODE_SET;
		lang->coverage_format = GDScriptLanguage::COVERAGE_FORMAT_LCOV;
	}

	~CoverageScopedReset() {
		if (!lang) {
			return;
		}
		lang->coverage_mode = saved_mode;
		lang->coverage_format = saved_format;
		lang->coverage_output_path = saved_output_path;
		lang->coverage_threshold = saved_threshold;
		lang->coverage_include = saved_include;
		lang->coverage_exclude = saved_exclude;
		lang->coverage_enabled = saved_enabled;
		lang->coverage_written = saved_written;
		lang->coverage_hits.clear();
		lang->coverage_func_hits.clear();
		lang->coverage_branch_hits.clear();
	}
};

TEST_SUITE("[Modules][GDScript]") {
	TEST_CASE("[Modules][GDScript] Coverage: configuration setters") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		SUBCASE("coverage_set_mode 'set' selects set mode") {
			lang->coverage_set_mode("set");
			CHECK(lang->coverage_mode == GDScriptLanguage::COVERAGE_MODE_SET);
		}

		SUBCASE("coverage_set_mode 'count' selects count mode") {
			lang->coverage_set_mode("count");
			CHECK(lang->coverage_mode == GDScriptLanguage::COVERAGE_MODE_COUNT);
		}

		SUBCASE("coverage_set_mode unknown value defaults to set") {
			lang->coverage_set_mode("bogus");
			CHECK(lang->coverage_mode == GDScriptLanguage::COVERAGE_MODE_SET);
		}

		SUBCASE("coverage_set_format 'lcov' selects LCOV") {
			lang->coverage_set_format("lcov");
			CHECK(lang->coverage_format == GDScriptLanguage::COVERAGE_FORMAT_LCOV);
		}

		SUBCASE("coverage_set_format 'cobertura' selects Cobertura") {
			lang->coverage_set_format("cobertura");
			CHECK(lang->coverage_format == GDScriptLanguage::COVERAGE_FORMAT_COBERTURA);
		}

		SUBCASE("coverage_set_format 'json' selects JSON") {
			lang->coverage_set_format("json");
			CHECK(lang->coverage_format == GDScriptLanguage::COVERAGE_FORMAT_JSON);
		}

		SUBCASE("coverage_set_format 'text' selects text") {
			lang->coverage_set_format("text");
			CHECK(lang->coverage_format == GDScriptLanguage::COVERAGE_FORMAT_TEXT);
		}

		SUBCASE("coverage_set_format unknown value defaults to LCOV") {
			lang->coverage_set_format("bogus");
			CHECK(lang->coverage_format == GDScriptLanguage::COVERAGE_FORMAT_LCOV);
		}

		SUBCASE("coverage_set_threshold stores the value") {
			lang->coverage_set_threshold(75.0f);
			CHECK(lang->coverage_threshold == doctest::Approx(75.0f));
		}

		SUBCASE("coverage_set_output stores the path") {
			lang->coverage_set_output("/tmp/test_cov.lcov");
			CHECK(lang->coverage_output_path == "/tmp/test_cov.lcov");
		}

		SUBCASE("coverage_add_include appends pattern") {
			lang->coverage_add_include("res://src/**");
			lang->coverage_add_include("res://lib/**");
			REQUIRE(lang->coverage_include.size() == 2);
			CHECK(lang->coverage_include[0] == "res://src/**");
			CHECK(lang->coverage_include[1] == "res://lib/**");
		}

		SUBCASE("coverage_add_exclude appends pattern") {
			lang->coverage_add_exclude("res://addons/**");
			REQUIRE(lang->coverage_exclude.size() == 1);
			CHECK(lang->coverage_exclude[0] == "res://addons/**");
		}
	}

	TEST_CASE("[Modules][GDScript] Coverage: start resets state") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		// Pre-populate data.
		lang->coverage_record_line("res://a.gd", 5);
		lang->coverage_record_func_entry("res://a.gd", "my_func");
		lang->coverage_record_branch("res://a.gd", 10, 42, true);
		lang->coverage_written = true;

		// start() must wipe everything and mark enabled.
		lang->coverage_start();

		CHECK(lang->coverage_hits.is_empty());
		CHECK(lang->coverage_func_hits.is_empty());
		CHECK(lang->coverage_branch_hits.is_empty());
		CHECK(lang->coverage_enabled);
		CHECK(!lang->coverage_written);
	}

	TEST_CASE("[Modules][GDScript] Coverage: record line (set mode)") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("set");
		lang->coverage_record_line("res://foo.gd", 10);

		REQUIRE(lang->coverage_hits.has("res://foo.gd"));
		const HashMap<int, int> &lines = lang->coverage_hits["res://foo.gd"];
		REQUIRE(lines.has(10));
		CHECK(lines[10] == 1);

		SUBCASE("second hit stays at 1 in set mode") {
			lang->coverage_record_line("res://foo.gd", 10);
			CHECK(lang->coverage_hits["res://foo.gd"][10] == 1);
		}
	}

	TEST_CASE("[Modules][GDScript] Coverage: record line (count mode)") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("count");
		lang->coverage_record_line("res://foo.gd", 10);
		lang->coverage_record_line("res://foo.gd", 10);
		lang->coverage_record_line("res://foo.gd", 10);

		REQUIRE(lang->coverage_hits.has("res://foo.gd"));
		CHECK(lang->coverage_hits["res://foo.gd"][10] == 3);
	}

	TEST_CASE("[Modules][GDScript] Coverage: record line include filter") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_add_include("res://src/**");
		lang->coverage_record_line("res://src/player.gd", 1);
		lang->coverage_record_line("res://addons/gut/gut.gd", 1);

		CHECK(lang->coverage_hits.has("res://src/player.gd"));
		CHECK(!lang->coverage_hits.has("res://addons/gut/gut.gd"));
	}

	TEST_CASE("[Modules][GDScript] Coverage: record line exclude filter") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_add_exclude("res://addons/**");
		lang->coverage_record_line("res://src/player.gd", 1);
		lang->coverage_record_line("res://addons/gut/gut.gd", 1);

		CHECK(lang->coverage_hits.has("res://src/player.gd"));
		CHECK(!lang->coverage_hits.has("res://addons/gut/gut.gd"));
	}

	TEST_CASE("[Modules][GDScript] Coverage: record func entry") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("count");
		lang->coverage_record_func_entry("res://bar.gd", "_ready");
		lang->coverage_record_func_entry("res://bar.gd", "_ready");
		lang->coverage_record_func_entry("res://bar.gd", "_process");

		REQUIRE(lang->coverage_func_hits.has("res://bar.gd"));
		const HashMap<String, int> &funcs = lang->coverage_func_hits["res://bar.gd"];
		CHECK(funcs["_ready"] == 2);
		CHECK(funcs["_process"] == 1);
	}

	TEST_CASE("[Modules][GDScript] Coverage: record func entry set mode caps at 1") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("set");
		lang->coverage_record_func_entry("res://bar.gd", "_ready");
		lang->coverage_record_func_entry("res://bar.gd", "_ready");

		CHECK(lang->coverage_func_hits["res://bar.gd"]["_ready"] == 1);
	}

	TEST_CASE("[Modules][GDScript] Coverage: record branch") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("count");
		// ip=100 is the branch instruction pointer, line=42 is the source line.
		lang->coverage_record_branch("res://baz.gd", 42, 100, true);
		lang->coverage_record_branch("res://baz.gd", 42, 100, true);
		lang->coverage_record_branch("res://baz.gd", 42, 100, false);

		REQUIRE(lang->coverage_branch_hits.has("res://baz.gd"));
		const HashMap<int, GDScriptLanguage::BranchResult> &branches = lang->coverage_branch_hits["res://baz.gd"];
		REQUIRE(branches.has(100));
		const GDScriptLanguage::BranchResult &br = branches[100];
		CHECK(br.taken == 2);
		CHECK(br.not_taken == 1);
		CHECK(br.line == 42);
	}

	TEST_CASE("[Modules][GDScript] Coverage: record branch set mode caps at 1") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_mode("set");
		lang->coverage_record_branch("res://baz.gd", 10, 50, true);
		lang->coverage_record_branch("res://baz.gd", 10, 50, true);
		lang->coverage_record_branch("res://baz.gd", 10, 50, false);

		const GDScriptLanguage::BranchResult &br = lang->coverage_branch_hits["res://baz.gd"][50];
		CHECK(br.taken == 1);
		CHECK(br.not_taken == 1);
	}

	TEST_CASE("[Modules][GDScript] Coverage: branch stores source line") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_record_branch("res://baz.gd", 99, 200, true);

		REQUIRE(lang->coverage_branch_hits.has("res://baz.gd"));
		REQUIRE(lang->coverage_branch_hits["res://baz.gd"].has(200));
		CHECK(lang->coverage_branch_hits["res://baz.gd"][200].line == 99);
	}

	TEST_CASE("[Modules][GDScript] Coverage: threshold check") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		// Record 4 lines, 3 hit and 1 not hit.
		lang->coverage_record_line("res://t.gd", 1);
		lang->coverage_record_line("res://t.gd", 2);
		lang->coverage_record_line("res://t.gd", 3);
		// Line 4 is coverable but not hit: inject it directly.
		lang->coverage_hits["res://t.gd"][4] = 0;

		SUBCASE("threshold 0.0 always passes") {
			lang->coverage_set_threshold(0.0f);
			CHECK(lang->coverage_check_threshold());
		}

		SUBCASE("threshold below actual coverage passes") {
			lang->coverage_set_threshold(70.0f); // actual is 75%
			CHECK(lang->coverage_check_threshold());
		}

		SUBCASE("threshold exactly at actual coverage passes") {
			lang->coverage_set_threshold(75.0f);
			CHECK(lang->coverage_check_threshold());
		}

		SUBCASE("threshold above actual coverage fails") {
			lang->coverage_set_threshold(76.0f); // actual is 75%
			CHECK(!lang->coverage_check_threshold());
		}
	}

	TEST_CASE("[Modules][GDScript] Coverage: threshold with no data passes") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_threshold(90.0f);
		CHECK(lang->coverage_check_threshold());
	}

	TEST_CASE("[Modules][GDScript] Coverage: summary string contains expected sections") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_record_line("res://summary_test.gd", 1);
		lang->coverage_record_func_entry("res://summary_test.gd", "_init");
		lang->coverage_record_branch("res://summary_test.gd", 5, 10, true);

		String s = lang->coverage_summary_string();

		CHECK_MESSAGE(s.contains("Lines"), "Summary must have a Lines column");
		CHECK_MESSAGE(s.contains("Funcs"), "Summary must have a Funcs column");
		CHECK_MESSAGE(s.contains("Branches"), "Summary must have a Branches column");
		CHECK_MESSAGE(s.contains("Total"), "Summary must have a Total row");
		CHECK_MESSAGE(s.contains("res://summary_test.gd"), "Summary must list the recorded file");
	}

	TEST_CASE("[Modules][GDScript] Coverage: summary string shows threshold status") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_record_line("res://t2.gd", 1);
		lang->coverage_hits["res://t2.gd"][2] = 0; // unhit line

		SUBCASE("pass indicator shown when threshold is met") {
			lang->coverage_set_threshold(40.0f);
			CHECK(lang->coverage_summary_string().contains(U"✓"));
		}

		SUBCASE("fail indicator shown when threshold is not met") {
			lang->coverage_set_threshold(90.0f);
			CHECK(lang->coverage_summary_string().contains(U"✗"));
		}

		SUBCASE("no threshold line when threshold is 0") {
			lang->coverage_set_threshold(0.0f);
			String s = lang->coverage_summary_string();
			CHECK(!s.contains("Threshold:"));
		}
	}

	TEST_CASE("[Modules][GDScript] Coverage: multiple files tracked independently") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_record_line("res://a.gd", 1);
		lang->coverage_record_line("res://a.gd", 2);
		lang->coverage_record_line("res://b.gd", 10);

		CHECK(lang->coverage_hits.has("res://a.gd"));
		CHECK(lang->coverage_hits.has("res://b.gd"));
		CHECK(lang->coverage_hits["res://a.gd"].size() == 2);
		CHECK(lang->coverage_hits["res://b.gd"].size() == 1);
	}

	TEST_CASE("[Modules][GDScript] Coverage: exclude takes priority over include") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_add_include("res://src/**");
		lang->coverage_add_exclude("res://src/ignored/**");

		lang->coverage_record_line("res://src/player.gd", 1);
		lang->coverage_record_line("res://src/ignored/stub.gd", 1);

		CHECK(lang->coverage_hits.has("res://src/player.gd"));
		CHECK(!lang->coverage_hits.has("res://src/ignored/stub.gd"));
	}
}

} // namespace GDScriptTests

#endif // TOOLS_ENABLED
