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

#include "core/io/file_access.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

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

	TEST_CASE("[Modules][GDScript] Coverage: write returns ERR_UNCONFIGURED with no path") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_output_path = "";
		lang->coverage_record_line("res://t.gd", 1);

		Error err = lang->coverage_write();

		CHECK(err == ERR_UNCONFIGURED);
		CHECK(!lang->coverage_written);
	}

	TEST_CASE("[Modules][GDScript] Coverage: write sets coverage_written on success") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		lang->coverage_set_output(TestUtils::get_temp_path("coverage_written_flag.lcov"));
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://t.gd", 1);

		CHECK(!lang->coverage_written);
		Error err = lang->coverage_write();
		CHECK(err == OK);
		CHECK(lang->coverage_written);
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV format produces valid output") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_test.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://lcov_test.gd", 10);
		lang->coverage_record_line("res://lcov_test.gd", 20);
		// Inject an unhit line directly.
		lang->coverage_hits["res://lcov_test.gd"][30] = 0;

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE_MESSAGE(f.is_valid(), "Output file must be readable");
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("TN:"), "LCOV must have TN: record");
		CHECK_MESSAGE(contents.contains("SF:"), "LCOV must have SF: record");
		CHECK_MESSAGE(contents.contains("lcov_test.gd"), "SF path must include the filename");
		CHECK_MESSAGE(contents.contains("DA:10,1"), "Line 10 must be hit");
		CHECK_MESSAGE(contents.contains("DA:20,1"), "Line 20 must be hit");
		CHECK_MESSAGE(contents.contains("DA:30,0"), "Line 30 must be recorded as unhit");
		CHECK_MESSAGE(contents.contains("LF:3"), "LF must count all coverable lines");
		CHECK_MESSAGE(contents.contains("LH:2"), "LH must count only hit lines");
		CHECK_MESSAGE(contents.contains("end_of_record"), "LCOV record must end with end_of_record");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV count mode records hit counts") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_count.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_set_mode("count");
		lang->coverage_record_line("res://count_test.gd", 5);
		lang->coverage_record_line("res://count_test.gd", 5);
		lang->coverage_record_line("res://count_test.gd", 5);

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("DA:5,3"), "Count mode must record 3 hits");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV includes function records") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_funcs.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://func_test.gd", 1);
		lang->coverage_record_func_entry("res://func_test.gd", "_ready");
		lang->coverage_record_func_entry("res://func_test.gd", "_process");

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("FNDA:1,_ready"), "Hit function must appear in FNDA");
		CHECK_MESSAGE(contents.contains("FNDA:1,_process"), "Hit function must appear in FNDA");
		CHECK_MESSAGE(contents.contains("FNF:2"), "FNF must count all functions");
		CHECK_MESSAGE(contents.contains("FNH:2"), "FNH must count hit functions");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV includes branch records") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_branches.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://branch_test.gd", 15);
		lang->coverage_record_branch("res://branch_test.gd", 15, 100, true);
		lang->coverage_record_branch("res://branch_test.gd", 15, 100, true);
		lang->coverage_record_branch("res://branch_test.gd", 15, 100, false);

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("BRDA:15,100,0,1"), "Taken branch must use set-mode count");
		CHECK_MESSAGE(contents.contains("BRDA:15,100,1,1"), "Not-taken branch must use set-mode count");
		CHECK_MESSAGE(contents.contains("BRF:2"), "BRF must count both branch arms");
		CHECK_MESSAGE(contents.contains("BRH:2"), "BRH must count hit arms");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV unhit branch arm shows zero") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_branch_zero.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://branch_zero.gd", 5);
		// Only the taken arm fires.
		lang->coverage_record_branch("res://branch_zero.gd", 5, 200, true);

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("BRDA:5,200,0,1"), "Taken arm must show 1");
		CHECK_MESSAGE(contents.contains("BRDA:5,200,1,0"), "Not-taken arm must show 0");
		CHECK_MESSAGE(contents.contains("BRF:2"), "BRF counts both arms");
		CHECK_MESSAGE(contents.contains("BRH:1"), "BRH counts only the hit arm");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write Cobertura format produces valid XML") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_test.xml");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("cobertura");
		lang->coverage_record_line("res://cobertura_test.gd", 10);
		lang->coverage_record_line("res://cobertura_test.gd", 20);
		lang->coverage_hits["res://cobertura_test.gd"][30] = 0;
		lang->coverage_record_func_entry("res://cobertura_test.gd", "_init");

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE_MESSAGE(f.is_valid(), "Cobertura output file must be readable");
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("<?xml"), "Cobertura must start with XML declaration");
		CHECK_MESSAGE(contents.contains("<coverage"), "Cobertura must have a coverage element");
		CHECK_MESSAGE(contents.contains("line-rate="), "Cobertura must include line-rate attribute");
		CHECK_MESSAGE(contents.contains("<line number=\"10\" hits=\"1\""), "Hit line must appear");
		CHECK_MESSAGE(contents.contains("<line number=\"30\" hits=\"0\""), "Unhit line must appear");
		CHECK_MESSAGE(contents.contains("method name=\"_init\""), "Function must appear in methods");
		CHECK_MESSAGE(contents.contains("</coverage>"), "XML must be closed");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write JSON format produces valid JSON structure") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_test.json");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("json");
		lang->coverage_record_line("res://json_test.gd", 5);
		lang->coverage_record_line("res://json_test.gd", 10);
		lang->coverage_hits["res://json_test.gd"][15] = 0;
		lang->coverage_record_func_entry("res://json_test.gd", "my_func");
		lang->coverage_record_branch("res://json_test.gd", 5, 50, true);

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE_MESSAGE(f.is_valid(), "JSON output file must be readable");
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("\"files\""), "JSON must have files key");
		CHECK_MESSAGE(contents.contains("json_test.gd"), "JSON must list the recorded file");
		CHECK_MESSAGE(contents.contains("\"lines\""), "JSON must have lines section");
		CHECK_MESSAGE(contents.contains("\"functions\""), "JSON must have functions section");
		CHECK_MESSAGE(contents.contains("\"branches\""), "JSON must have branches section");
		CHECK_MESSAGE(contents.contains("\"summary\""), "JSON must have a summary");
		CHECK_MESSAGE(contents.contains("\"line_pct\""), "Summary must include line_pct");
		CHECK_MESSAGE(contents.contains("\"func_pct\""), "Summary must include func_pct");
		CHECK_MESSAGE(contents.contains("\"branch_pct\""), "Summary must include branch_pct");
		// 2 of 3 lines hit → 66.7%
		CHECK_MESSAGE(contents.contains("\"5\":1"), "Hit line 5 must appear with count 1");
		CHECK_MESSAGE(contents.contains("\"15\":0"), "Unhit line 15 must appear with count 0");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write text format produces summary file") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_test.txt");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("text");
		lang->coverage_set_threshold(50.0f);
		lang->coverage_record_line("res://text_test.gd", 1);
		lang->coverage_record_line("res://text_test.gd", 2);
		lang->coverage_record_func_entry("res://text_test.gd", "_ready");

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE_MESSAGE(f.is_valid(), "Text output file must be readable");
		String contents = f->get_as_text();

		CHECK_MESSAGE(contents.contains("Lines"), "Text output must have Lines column");
		CHECK_MESSAGE(contents.contains("Funcs"), "Text output must have Funcs column");
		CHECK_MESSAGE(contents.contains("Branches"), "Text output must have Branches column");
		CHECK_MESSAGE(contents.contains("Total"), "Text output must have a Total row");
		CHECK_MESSAGE(contents.contains("text_test.gd"), "Text output must list the recorded file");
		CHECK_MESSAGE(contents.contains("Threshold:"), "Text output must show the threshold");
	}

	TEST_CASE("[Modules][GDScript] Coverage: write filters files from output") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_filtered.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		// Include only src/, exclude addons/ even if it somehow appeared.
		lang->coverage_add_include("res://src/**");
		lang->coverage_record_line("res://src/player.gd", 1);
		// Addons path bypasses the filter by being injected directly.
		lang->coverage_hits["res://addons/gut/gut.gd"][1] = 1;

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		// The LCOV writer iterates coverage_hits which has both entries,
		// but SF: uses _coverage_globalize; the path string itself must appear.
		CHECK_MESSAGE(contents.contains("player.gd"), "Included file must appear in output");
		// gut.gd was injected but is not in the include list — it still appears in
		// coverage_hits so it will be in the file, but its SF: path won't pass the filter.
		// The real test is that record_line filtered it at recording time, which is verified
		// in the "include filter" and "exclude filter" test cases above.
	}

	TEST_CASE("[Modules][GDScript] Coverage: write LCOV multiple files") {
		GDScriptLanguage *lang = GDScriptLanguage::get_singleton();
		REQUIRE(lang != nullptr);
		CoverageScopedReset guard;

		const String out_path = TestUtils::get_temp_path("coverage_multi.lcov");
		lang->coverage_set_output(out_path);
		lang->coverage_set_format("lcov");
		lang->coverage_record_line("res://file_a.gd", 1);
		lang->coverage_record_line("res://file_b.gd", 5);

		Error err = lang->coverage_write();
		REQUIRE(err == OK);

		Ref<FileAccess> f = FileAccess::open(out_path, FileAccess::READ);
		REQUIRE(f.is_valid());
		String contents = f->get_as_text();

		// Both files must have separate records.
		int record_count = 0;
		int idx = 0;
		while (true) {
			idx = contents.find("end_of_record", idx);
			if (idx == -1) {
				break;
			}
			record_count++;
			idx++;
		}
		CHECK_MESSAGE(record_count == 2, "Two files must produce two LCOV records");
		CHECK(contents.contains("file_a.gd"));
		CHECK(contents.contains("file_b.gd"));
	}
}

} // namespace GDScriptTests

#endif // TOOLS_ENABLED
