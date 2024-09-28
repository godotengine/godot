/**************************************************************************/
/*  test_string.h                                                         */
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
#include <iostream>
#include <chrono>

namespace TestFuzzySearch {

double calculateMean(const Vector<int>& numbers) {
    double sum = 0.0;
    for(int num : numbers) {
        sum += num;
    }
    return sum / numbers.size();
}

// Function to calculate standard deviation
double calculateStdDev(const Vector<int>& numbers) {
    double mean = calculateMean(numbers);
    double variance = 0.0;

    for(int num : numbers) {
        variance += (num - mean) * (num - mean);
    }
    variance /= numbers.size();  // Population standard deviation formula
    return std::sqrt(variance);
}

auto bench(String query, String dataset_path, String expected_result, String algorithm) {
	Ref<FileAccess> some_project_dir_tree = FileAccess::open(TestUtils::get_data_path("fuzzy_search/" + dataset_path), FileAccess::READ);
	REQUIRE(!some_project_dir_tree.is_null());

	auto data = some_project_dir_tree->get_as_utf8_string().split("\n");
	CHECK(data.size() > 0);


	Vector<int> results;
	String top_result;

	// run twice for a warmp up
	for(int i = 0; i < 2; i++) {
		results.clear();
		
		for(int j = 0; j < 10; j++) {
			auto start = std::chrono::high_resolution_clock::now();

			Ref<FuzzySearch> fuzzySearch{};

			Vector<Ref<FuzzySearchResult>> res;
			if(algorithm == "lev") {
				res = fuzzySearch->search_all_lev(query, data);
			}
			else {
				res = fuzzySearch->search_all(query, data);
			}

			auto end = std::chrono::high_resolution_clock::now();

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			results.push_back(duration);

			if(res.size() > 0) {
				top_result = res[0]->target;
			}
		}
	}

	MESSAGE(algorithm, ",", query, ",", dataset_path, ",", calculateStdDev(results), ",", calculateMean(results), ",", top_result);
}

TEST_CASE("[FuzzySearch] Find Stuff") {
	Ref<FileAccess> tests = FileAccess::open(TestUtils::get_data_path("fuzzy_search/fuzzy_search_tests.txt"), FileAccess::READ);
	REQUIRE(!tests.is_null());

	while(true) {
		auto line = tests->get_csv_line();
		if(line.is_empty()) {
			break;
		}

		if(line.size() != 3) {
			break;
		}

		auto query = line[0];
		auto dataset_path = line[1];
		auto expected_result = line[2];

		bench(query, dataset_path, expected_result, "fzf");
		bench(query, dataset_path, expected_result, "lev");
	}
}

} // namespace TestString

#endif // TEST_FUZZY_SEARCH_H
