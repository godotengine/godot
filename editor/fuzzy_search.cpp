/**************************************************************************/
/*  fuzzy_search.cpp                                                      */
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

#include "fuzzy_search.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "modules/regex/regex.h"

void FuzzySearch::set_query(const String &p_query) {
	Ref<RegEx> split_query_regex = RegEx::create_from_string("[^\\/\\s]+");
	TypedArray<RegExMatch> query_tokens = split_query_regex->search_all(p_query);

	for (int i = 0; i < query_tokens.size(); i++) {
		Ref<RegExMatch> query_token = query_tokens[i];

		PackedStringArray strings = query_token->get_strings();
		if (strings.is_empty()) {
			continue;
		}

		m_query_tokens.append(strings[0]);
	}
}

const Vector<FuzzySearchResult> &FuzzySearch::commit() {
	if (m_query_tokens.is_empty()) {
		return m_results;
	}

	struct FuzzySearchResultComparator {
		bool operator()(const FuzzySearchResult &A, const FuzzySearchResult &B) const {
			return A.score > B.score;
		}
	};

	SortArray<FuzzySearchResult, FuzzySearchResultComparator> sorter;
	sorter.sort(m_results.ptrw(), m_results.size());

	int mean_score = m_total_score / m_results.size();

	int i = 0;
	while (m_results[i].score > mean_score && i < m_results.size() && i < 300) {
		i++;
	}

	m_results.resize(i);

	return m_results;
}

String FuzzySearch::decorate(const FuzzySearchResult &p_result) {
	if (p_result.target.is_empty()) {
		return p_result.target;
	}

	if (p_result.matches.is_empty()) {
		return p_result.target;
	}

	String result;
	for (int i = 0; i < p_result.target.size(); i++) {
		result += p_result.target[i];
		if (p_result.matches.has(i)) {
			result += U'\u0332';
		}
	}

	// + " " + itos(p_result.score)

	return result;
}

FuzzySearchResult FuzzySearch::fuzzy_search(const String &p_query, const String &p_target, int position_offset) {
	if (p_query.is_empty()) {
		return {};
	}

	if (p_target.is_empty()) {
		return {};
	}

	// Convert strings to lowercase
	String str1 = p_query.to_lower();
	String str2 = p_target.to_lower();

	int lenStr1 = str1.length();
	int lenStr2 = str2.length();

	int n = lenStr1 + 1;
	int m = lenStr2 + 1;
	int beginIdx = m + 1;

	// Create a flat array to store the edit distances
	Vector<int> dp;
	// And another array to store the counts of sequences that match
	Vector<int> matches;
	for (int i = 0; i < n * m; i++) {
		dp.append(0);
		matches.append(0);
	}

	// Calculate the edit distances
	for (int i = 1; i < n; i++) {
		for (int j = 1; j < m; j++) {
			int currentIdx = i * m + j;
			int northWestIdx = currentIdx - (m + 1);
			int westIdx = currentIdx - 1;

			int score = 0;

			if (dp[northWestIdx] == 0 && i > 1) {
				score = 0;
			} else if (str1[i - 1] == str2[j - 1]) {
				score = 1;

				// Boost score if beginging of word
				if (currentIdx == beginIdx) {
					score += 8;
				}

				matches.set(currentIdx, matches[northWestIdx] + 1);

				// Boost score if we're on a match streak
				score += matches[currentIdx] * 5;
			}

			if (score && (dp[northWestIdx] + score) >= dp[westIdx]) {
				score = score + dp[northWestIdx];
			} else {
				score = dp[westIdx];
			}

			dp.set(currentIdx, score);
		}
	}

	// The bottom-right cell of the matrix contains the Levenshtein distance
	int mostSouthWestIdx = lenStr2 * (lenStr1 + 1) + lenStr1;

	FuzzySearchResult res;
	res.score = dp[mostSouthWestIdx];

	if (res.score == 0) {
		return res;
	}

	int p = mostSouthWestIdx;

	// Walk back through the matches and get all matched chars.
	// Useful for highlighting match characters in search result UX.
	while (p > 0) {
		if (matches[p] > 0) {
			res.matches.insert(((p % m) - 1) + position_offset);
			p -= m;
		}

		p -= 1;
	}

	return res;
}

FuzzySearchResult FuzzySearch::fuzzy_search_path_components(const String &p_query_token, const PackedStringArray &p_path_components) {
	FuzzySearchResult result;
	int offset = 0;

	for (int i = 0; i < p_path_components.size(); i++) {
		bool end_of_path = i == p_path_components.size() - 1;

		FuzzySearchResult res = fuzzy_search(p_query_token, p_path_components[i], offset);
		if (end_of_path) {
			res.score *= 100;
		}

		if (res.score > result.score) {
			result = res;
		}

		offset += p_path_components[i].length() + 1;
	}

	return result;
}

void FuzzySearch::fuzzy_search_path(const String &p_path) {
	FuzzySearchResult result;
	result.target = p_path;

	if (!m_query_tokens.is_empty()) {
		PackedStringArray target = p_path.to_lower().split("/");

		for (int i = 0; i < m_query_tokens.size(); i++) {
			FuzzySearchResult res = fuzzy_search_path_components(m_query_tokens[i], target);

			result.score += res.score;

			for (int match : res.matches) {
				result.matches.insert(match);
			}
		}

		m_total_score += result.score;
	}

	m_results.push_back(result);
}
