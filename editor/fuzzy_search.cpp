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
#include "scene/gui/tree.h"

const int max_results = 100;

Vector<int> FuzzySearchResult::get_matches() const {
	Vector<int> matches_array;

	for (int i : matches) {
		matches_array.push_back(i);
	}

	matches_array.sort();

	return matches_array;
}

const Vector<int> &FuzzySearchResult::get_matches_as_substr_sequences() const {
	if (matches.is_empty()) {
		return m_matches_as_substr_sequences_cache;
	}

	// Considering this function may be called every scroll of a search list;
	// better to cache this.
	if (!m_matches_as_substr_sequences_cache.is_empty()) {
		return m_matches_as_substr_sequences_cache;
	}

	Vector<int> matches_array = get_matches();

	for (int i = 0; i < matches_array.size(); i++) {
		int begin = i;
		while ((i < matches.size() - 1) && (matches_array[i] + 1) == matches_array[i + 1]) {
			i++;
		}

		m_matches_as_substr_sequences_cache.push_back(matches_array[begin]);
		m_matches_as_substr_sequences_cache.push_back(matches_array[i] - matches_array[begin] + 1);
	}

	return m_matches_as_substr_sequences_cache;
}

PackedStringArray get_query(const String &p_query) {
	PackedStringArray query_tokens;

	Ref<RegEx> split_query_regex = RegEx::create_from_string("[^\\/\\s]+");
	TypedArray<RegExMatch> query_token_matches = split_query_regex->search_all(p_query);

	for (int i = 0; i < query_token_matches.size(); i++) {
		Ref<RegExMatch> query_token_match = query_token_matches[i];

		PackedStringArray strings = query_token_match->get_strings();
		if (strings.is_empty()) {
			continue;
		}

		query_tokens.append(strings[0]);
	}

	return query_tokens;
}

Vector<Ref<FuzzySearchResult>> sort_and_filter(const Vector<Ref<FuzzySearchResult>> &p_results) {
	Vector<Ref<FuzzySearchResult>> res;

	float total_score = 0;
	for (const Ref<FuzzySearchResult> &result : p_results) {
		total_score += result->score;
	}

	float mean_score = total_score / p_results.size();

	struct FuzzySearchResultComparator {
		bool operator()(const Ref<FuzzySearchResult> &A, const Ref<FuzzySearchResult> &B) const {
			return A->score > B->score;
		}
	};

	// Prune low score entries before even sorting
	for (Ref<FuzzySearchResult> i : p_results) {
		if (i->score >= mean_score * 0.5) {
			res.push_back(i);
		}

		if (res.size() > max_results) {
			break;
		}
	}

	SortArray<Ref<FuzzySearchResult>, FuzzySearchResultComparator> sorter;
	sorter.sort(res.ptrw(), res.size());

	return res;
}

Ref<FuzzySearchResult> fuzzy_search(const String &p_query, const String &p_target, int position_offset) {
	if (p_query.is_empty()) {
		return nullptr;
	}

	if (p_target.is_empty()) {
		return nullptr;
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

	if (dp[mostSouthWestIdx] == 0) {
		return nullptr;
	}

	Ref<FuzzySearchResult> res;
	res.instantiate();
	res->score = dp[mostSouthWestIdx];

	int p = mostSouthWestIdx;

	// Walk back through the matches and get all matched chars.
	// Useful for highlighting match characters in search result UX.
	while (p > 0) {
		if (matches[p] > 0) {
			res->matches.insert(((p % m) - 1) + position_offset);
			p -= m;
		}

		p -= 1;
	}

	return res;
}

// Iterate over every component in the path and use the highest scoring match as the result.
// Also weights the final component's score massively heaiver considering people tend to search for files, not directories.
Ref<FuzzySearchResult> fuzzy_search_path_components(const String &p_query_token, const PackedStringArray &p_path_components) {
	Ref<FuzzySearchResult> result;

	int offset = 0;

	for (int i = 0; i < p_path_components.size(); i++) {
		bool end_of_path = i == p_path_components.size() - 1;

		Ref<FuzzySearchResult> res = fuzzy_search(p_query_token, p_path_components[i], offset);
		offset += p_path_components[i].length() + 1;

		if (res.is_null() || res->score == 0) {
			continue;
		}

		if (end_of_path) {
			res->score *= 100;
		}

		if (result.is_null()) {
			result = res;
		} else if (res->score > result->score) {
			result = res;
		}
	}

	return result;
}

Ref<FuzzySearchResult> fuzzy_search_path(const PackedStringArray &p_query_tokens, const String &p_path) {
	Ref<FuzzySearchResult> result;

	PackedStringArray target = p_path.to_lower().split("/");

	for (const String &query_token : p_query_tokens) {
		Ref<FuzzySearchResult> res = fuzzy_search_path_components(query_token, target);
		if (res.is_null()) {
			return nullptr;
		}

		if (result.is_null()) {
			result = res;
			result->target = p_path;
		} else {
			result->score += res->score;
		}

		for (int match : res->matches) {
			result->matches.insert(match);
		}
	}

	return result;
}

Vector<Ref<FuzzySearchResult>> FuzzySearch::search_all(const String &p_query_tokens, const PackedStringArray &p_search_data) {
	Vector<Ref<FuzzySearchResult>> res;

	// Just spit out the results list if no query is given.
	if (p_query_tokens.is_empty()) {
		for (int i = 0; (i < max_results) && (i < p_search_data.size()); i++) {
			Ref<FuzzySearchResult> r;
			r.instantiate();
			r->target = p_search_data[i];
			res.push_back(r);
		}

		return res;
	}

	PackedStringArray query_tokens = get_query(p_query_tokens);

	for (const String &search_line : p_search_data) {
		Ref<FuzzySearchResult> r = fuzzy_search_path(query_tokens, search_line);
		if (!r.is_null()) {
			res.append(r);
		}
	}

	return sort_and_filter(res);
}

/*
// old search function
// maybe should leave this as a config option
float score_path(const String &p_search, const String &p_path) {
	if(!p_search.is_subsequence_ofn(p_path)) {
		return 0.0;
	}

	float score = 0.9f + .1f * (p_search.length() / (float)p_path.length());

	// Exact match.
	if (p_search == p_path) {
		return 1.2f;
	}

	// Positive bias for matches close to the beginning of the file name.
	String file = p_path.get_file();
	int pos = file.findn(p_search);
	if (pos != -1) {
		return score * (1.0f - 0.1f * (float(pos) / file.length()));
	}

	// Similarity
	return p_path.to_lower().similarity(p_search.to_lower());
}


Vector<Ref<FuzzySearchResult>> FuzzySearch::search_all(const String& p_query_tokens, const PackedStringArray &p_search_data) {
	Vector<Ref<FuzzySearchResult>> res;

	PackedStringArray query_tokens = get_query(p_query_tokens);

	for(const String& search_line : p_search_data) {
		float score = score_path(p_query_tokens, search_line) * 100.0;
		if(score > 0) {
			Ref<FuzzySearchResult> r;
			r.instantiate();
			r->target = search_line;
			r->score = score;
			res.append(r);
		}
	}

	sort_and_filter(res);

	return res;
}
*/

void FuzzySearch::draw_matches(Tree *p_tree) {
	if (p_tree == nullptr) {
		return;
	}

	TreeItem *head = p_tree->get_root();
	if (head == nullptr) {
		return;
	}

	Ref<Font> font = p_tree->get_theme_font("font");
	if (!font.is_valid()) {
		return;
	}

	int font_size = p_tree->get_theme_font_size("font_size");

	Vector2 margin_and_scroll_offset = -p_tree->get_scroll();
	margin_and_scroll_offset.x += p_tree->get_theme_constant("item_margin");
	margin_and_scroll_offset.y += font->get_string_size("A", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size).y;

	Vector2 magic_numbers = Vector2(23, -5);
	margin_and_scroll_offset += magic_numbers;

	Ref<Texture2D> icon = head->get_icon(0);
	if (icon.is_valid()) {
		margin_and_scroll_offset.x += icon->get_width();
	}

	while (head != nullptr && head->is_visible()) {
		Ref<FuzzySearchResult> fuzzy_search_result = head->get_metadata(0);
		if (fuzzy_search_result.is_valid()) {
			const Vector<int> &substr_sequences = fuzzy_search_result->get_matches_as_substr_sequences();

			for (int i = 0; i < substr_sequences.size(); i += 2) {
				String str_left_of_match = fuzzy_search_result->target.substr(0, substr_sequences[i]);
				String match_str = fuzzy_search_result->target.substr(substr_sequences[i], substr_sequences[i + 1]);

				Vector2 position = font->get_string_size(str_left_of_match, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size);
				position.y = 0;
				position += p_tree->get_item_rect(head, 0).position;
				position += margin_and_scroll_offset;

				Vector2 size = font->get_string_size(match_str, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size);

				p_tree->draw_rect(Rect2(position, size), Color(1, 1, 1, 0.07), true);
				p_tree->draw_rect(Rect2(position, size), Color(0.5, 0.7, 1.0, 0.4), false, 1);
			}
		}

		head = head->get_next_visible();
	}
}
