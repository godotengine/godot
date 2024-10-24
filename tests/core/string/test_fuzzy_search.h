/**************************************************************************/
/*  test_fuzzy_search.h                                                   */
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

#ifndef TEST_FUZZY_SEARCH_H
#define TEST_FUZZY_SEARCH_H

#include "core/string/fuzzy_search.h"
#include "tests/test_macros.h"
#include <chrono>
#include <iostream>

namespace TestFuzzySearch {

struct FuzzySearchTestCase {
	String query;
	String expected;
};

struct FuzzySearchTestOutcome {
	String top_result;
	int result_count;
};

struct FuzzySearchBenchmarkResult {
	double average_ms;
	double std_dev_ms;
	FuzzySearchTestOutcome outcome;
};

// Ideally each of these test queries should represent a different aspect, and potentially bottleneck, of the search process.
const FuzzySearchTestCase test_cases[] = {
	// Short query, many matches, few adjacent characters
	{ "///gd", "./menu/hud/hud.gd" },
	// Filename match with typo
	{ "sm.png", "./entity/blood_sword/sam.png" },
	// Multipart filename word matches
	{ "ham ", "./entity/game_trap/ha_missed_me.wav" },
	// Single word token matches
	{ "push background", "./entity/background_zone1/background/push.png" },
	// Long token matches
	{ "background_freighter background png", "./entity/background_freighter/background/background.png" },
	// Many matches, many short tokens
	{ "menu menu characters wav", "./menu/menu/characters/smoker/0.wav" },
	// Maximize total matches
	{ "entity gd", "./entity/entity_man.gd" }
};

double calculate_mean(const Vector<double> &p_numbers) {
	double sum = 0.0;
	for (double num : p_numbers) {
		sum += num;
	}
	return sum / static_cast<double>(p_numbers.size());
}

double calculate_std_dev(const Vector<double> &p_numbers) {
	double mean = calculate_mean(p_numbers);
	double variance = 0.0;
	for (double num : p_numbers) {
		variance += (num - mean) * (num - mean);
	}
	variance /= static_cast<double>(p_numbers.size());
	return std::sqrt(variance);
}

auto load_test_data(int p_repeat = 1) {
	// This file has 1k entries so p_repeat can be used to benchmark in multiples of 1k
	Ref<FileAccess> fp = FileAccess::open(TestUtils::get_data_path("fuzzy_search/project_dir_tree.txt"), FileAccess::READ);
	REQUIRE(!fp.is_null());
	auto lines = fp->get_as_utf8_string().split("\n");
	Vector<String> all_lines;
	while (p_repeat-- > 0) {
		all_lines.append_array(lines);
	}
	CHECK(lines.size() > 0);
	return all_lines;
}

FuzzySearchTestOutcome get_top_result_and_count(String &p_query, Vector<String> &p_lines, int p_max_results = 100) {
	FuzzySearch search;
	search.set_query(p_query);
	search.max_results = p_max_results;
	Vector<FuzzySearchResult> results;
	search.search_all(p_lines, results);
	return { results.size() > 0 ? results[0].target : "<no result>", (int)results.size() };
}

FuzzySearchBenchmarkResult bench(String p_query, Vector<String> p_targets) {
	Vector<double> timings;
	FuzzySearchTestOutcome result;

	// run twice for a warmp up
	for (int i = 0; i < 2; i++) {
		timings.clear();

		for (int j = 0; j < 10; j++) {
			auto start = std::chrono::high_resolution_clock::now();
			result = get_top_result_and_count(p_query, p_targets);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			timings.push_back(duration / 1000.0f); // Convert to fractional ms
		}
	}

	return { calculate_mean(timings), calculate_std_dev(timings), result };
}

/*
TEST_CASE("[Stress][FuzzySearch] Benchmark fuzzy search") {
	auto targets = load_test_data(20);
	print_line(vformat("Benchmarking fuzzy search against %dk targets", targets.size() / 1000));
	print_line("Query\tMean (ms)\tStd Dev (ms)\tMatches");
	int i = 1;
	for (auto test_case : test_cases) {
		auto result = bench(test_case.query, targets);
		print_line(vformat("%d\t%4.2f\t\t%4.2f\t\t%d", i++, result.average_ms, result.std_dev_ms, result.outcome.result_count));
	}
}
*/

TEST_CASE("[FuzzySearch] Test fuzzy search results") {
	auto targets = load_test_data();
	for (auto test_case : test_cases) {
		CHECK_EQ(get_top_result_and_count(test_case.query, targets).top_result, test_case.expected);
	}
}

} //namespace TestFuzzySearch

#endif // TEST_FUZZY_SEARCH_H
