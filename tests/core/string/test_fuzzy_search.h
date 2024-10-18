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

double calculate_mean(const Vector<double> &p_numbers) {
	double sum = 0.0;
	for (double num : p_numbers) {
		sum += num;
	}
	return sum / (double)p_numbers.size();
}

// Function to calculate standard deviation
double calculate_std_dev(const Vector<double> &p_numbers) {
	double mean = calculate_mean(p_numbers);
	double variance = 0.0;

	for (double num : p_numbers) {
		variance += (num - mean) * (num - mean);
	}
	variance /= (double)p_numbers.size(); // Population standard deviation formula
	return std::sqrt(variance);
}

auto load_test_cases() {
	Ref<FileAccess> tests = FileAccess::open(TestUtils::get_data_path("fuzzy_search/fuzzy_search_tests.txt"), FileAccess::READ);
	REQUIRE(!tests.is_null());

	Vector<FuzzySearchTestCase> test_cases;
	while (true) {
		auto line = tests->get_csv_line();
		if (line.size() != 2) {
			break;
		}
		test_cases.append({ line[0], line[1] });
	}
	return test_cases;
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

auto get_top_result(String &p_query, Vector<String> &p_lines) {
	Vector<Ref<FuzzySearchResult>> res = FuzzySearch::search_all(p_query, p_lines);
	if (res.size() > 0) {
		return res[0]->target;
	}
	return String("<no result>");
}

auto bench(String p_query, Vector<String> p_targets) {
	Vector<double> timings;

	// run twice for a warmp up
	for (int i = 0; i < 2; i++) {
		timings.clear();

		for (int j = 0; j < 10; j++) {
			auto start = std::chrono::high_resolution_clock::now();
			get_top_result(p_query, p_targets);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			timings.push_back(duration / 1000.0); // Convert to ms
		}
	}

	MESSAGE(vformat("%-15s\t%4.2f\t\t%4.2f", p_query, calculate_mean(timings), calculate_std_dev(timings)));
}

/*
TEST_CASE("[Stress][FuzzySearch] Benchmark fuzzy search") {
	auto targets = load_test_data(20);
	MESSAGE("Query\t\tMean (ms)\tStd Dev (ms)\tTargets: ", targets.size());
	for (auto test_case : load_test_cases()) {
		bench(test_case.query, targets);
	}
}
*/

TEST_CASE("[FuzzySearch] Test fuzzy search results") {
	auto targets = load_test_data();
	for (auto test_case : load_test_cases()) {
		CHECK_EQ(get_top_result(test_case.query, targets), test_case.expected);
	}
}

} //namespace TestFuzzySearch

#endif // TEST_FUZZY_SEARCH_H
