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
#include "scene/gui/tree.h"

const float cull_factor = 0.1f;
const float cull_cutoff = 30.0f;
const String boundary_chars = "/\\-_.";

bool is_valid_interval(Vector2i p_interval) {
	// Empty intervals are represented as (-1, -1).
	return p_interval.x >= 0 && p_interval.y >= p_interval.x;
}

Vector2i extend_interval(Vector2i p_a, Vector2i p_b) {
	if (!is_valid_interval(p_a)) {
		return p_b;
	}
	if (!is_valid_interval(p_b)) {
		return p_a;
	}
	return Vector2(MIN(p_a.x, p_b.x), MAX(p_a.y, p_b.y));
}

Ref<FuzzyTokenMatch> new_token_match() {
	Ref<FuzzyTokenMatch> match;
	match.instantiate();
	return match;
}

Ref<FuzzySearchResult> new_search_result() {
	Ref<FuzzySearchResult> result;
	result.instantiate();
	return result;
}

bool is_word_boundary(const String &str, const int index) {
	if (index == -1 || index == str.size()) {
		return true;
	}
	return boundary_chars.find_char(str[index]) >= 0;
}

void FuzzyTokenMatch::add_substring(const int substring_start, const int substring_length) {
	substrings.append(Vector2i(substring_start, substring_length));
	matched_length += substring_length;
	int substring_end = substring_start + substring_length - 1;
	interval = extend_interval(interval, Vector2i(substring_start, substring_end));
}

bool FuzzyTokenMatch::intersects(const Vector2i other_interval) {
	if (!is_valid_interval(interval) || !is_valid_interval(other_interval)) {
		return false;
	}
	return interval.y >= other_interval.x && interval.x <= other_interval.y;
}

bool FuzzySearchResult::can_add_token_match(Ref<FuzzyTokenMatch> &p_match) {
	if (p_match.is_null() || p_match->misses() > miss_budget) {
		return false;
	}

	if (p_match->intersects(match_interval)) {
		if (token_matches.size() == 1) {
			return false;
		}
		for (Ref<FuzzyTokenMatch> existing_match : token_matches) {
			if (existing_match->intersects(p_match->interval)) {
				return false;
			}
		}
	}

	return true;
}

void FuzzySearchResult::score_token_match(Ref<FuzzyTokenMatch> &p_match) {
	// This can always be tweaked more. The intuition is that exact matches should almost always
	// be prioritized over broken up matches, and other criteria more or less act as tie breakers.

	p_match->score = -20 * p_match->misses();

	for (Vector2i substring : p_match->substrings) {
		// Score longer substrings higher than short substrings
		int substring_score = substring.y * substring.y;
		// Score matches deeper in path higher than shallower matches
		if (substring.x > dir_index) {
			substring_score *= 2;
		}
		// Score matches on a word boundary higher than matches within a word
		if (is_word_boundary(target, substring.x - 1) || is_word_boundary(target, substring.x + substring.y)) {
			substring_score += 4;
		}
		// Score exact query matches higher than non-compact subsequence matches
		if (substring.y == p_match->token_length) {
			substring_score += 100;
		}
		p_match->score += substring_score;
	}
}

void FuzzySearchResult::add_token_match(Ref<FuzzyTokenMatch> &p_match) {
	score += p_match->score;
	match_interval = extend_interval(match_interval, p_match->interval);
	miss_budget -= p_match->misses();
	token_matches.append(p_match);
}

Vector<Ref<FuzzySearchResult>> FuzzySearch::sort_and_filter(const Vector<Ref<FuzzySearchResult>> &p_results) {
	Vector<Ref<FuzzySearchResult>> results;

	if (p_results.is_empty()) {
		return results;
	}

	float avg_score = 0;
	float max_score = 0;

	for (const Ref<FuzzySearchResult> &result : p_results) {
		avg_score += result->score;
		if (result->score > max_score) {
			max_score = result->score;
		}
	}

	// TODO: Tune scoring and culling here to display fewer subsequence soup matches when good matches
	//  are available.
	avg_score /= p_results.size();
	float cull_score = MIN(cull_cutoff, Math::lerp(avg_score, max_score, cull_factor));

	struct FuzzySearchResultComparator {
		bool operator()(const Ref<FuzzySearchResult> &A, const Ref<FuzzySearchResult> &B) const {
			// Sort on (score, length, alphanumeric) to ensure consistent ordering
			if (A->score == B->score) {
				if (A->target.length() == B->target.length()) {
					return A->target < B->target;
				}
				return A->target.length() < B->target.length();
			}
			return A->score > B->score;
		}
	};

	// Prune low score entries before sorting
	for (Ref<FuzzySearchResult> i : p_results) {
		if (i->score >= cull_score) {
			results.push_back(i);
		}
	}

	SortArray<Ref<FuzzySearchResult>, FuzzySearchResultComparator> sorter;

	if (results.size() > max_results) {
		sorter.partial_sort(0, results.size(), max_results, results.ptrw());
		results.resize(max_results);
	} else {
		sorter.sort(results.ptrw(), results.size());
	}

	return results;
}

void FuzzySearch::reset_match(Ref<FuzzyTokenMatch> &p_match, const String &p_token) {
	p_match->score = 0;
	p_match->token_length = p_token.length();
	p_match->matched_length = 0;
	p_match->interval = Vector2i(-1, -1);
	p_match->substrings.clear();
}

void FuzzySearch::reset_result(Ref<FuzzySearchResult> &p_result, const String &p_target) {
	p_result->score = 0;
	p_result->target = p_target;
	p_result->dir_index = p_target.rfind_char('/');
	p_result->miss_budget = max_misses;
	p_result->match_interval = Vector2i(-1, -1);
	p_result->token_matches.clear();
}

bool FuzzySearch::try_match_token(
		Ref<FuzzyTokenMatch> p_match,
		const String &p_token,
		const String &p_target,
		int p_offset,
		int p_miss_budget) {
	reset_match(p_match, p_token);
	int run_start = -1;
	int run_len = 0;

	if (!allow_subsequences) {
		int idx = p_target.find(p_token, p_offset);
		if (idx == -1) {
			return false;
		}
		p_match->add_substring(idx, p_token.length());
		return true;
	}

	// Search for the subsequence p_token in p_target starting from p_offset, recording each substring for
	// later scoring and display.
	for (int i = 0; i < p_token.length(); i++) {
		int new_offset = p_target.find_char(p_token[i], p_offset);
		if (new_offset < 0) {
			if (--p_miss_budget < 0) {
				return false;
			}
		} else {
			if (run_start == -1 || p_offset != new_offset) {
				if (run_start != -1) {
					p_match->add_substring(run_start, run_len);
				}
				run_start = new_offset;
				run_len = 1;
			} else {
				run_len += 1;
			}
			p_offset = new_offset + 1;
		}
	}

	if (run_start != -1) {
		p_match->add_substring(run_start, run_len);
	}

	return true;
}

bool FuzzySearch::fuzzy_search(Ref<FuzzySearchResult> p_result, const String &p_target) {
	if (p_result.is_null() || p_target.is_empty()) {
		return false;
	}

	reset_result(p_result, p_target);

	String adjusted_target = case_sensitive ? p_target : p_target.to_lower();
	Ref<FuzzyTokenMatch> match = new_token_match();

	// For each token, eagerly generate subsequences starting from index 0 and keep the best scoring one
	// which does not conflict with prior token matches. This is not ensured to find the highest scoring
	// combination of matches, or necessarily the highest scoring single subsequence, as it only considers
	// eager subsequences for a given index, and likewise eagerly finds matches for each token in sequence.
	for (const String &token : tokens) {
		int offset = 0;
		// TODO : Consider avoiding the FuzzyTokenMatch allocation by either passing a reference to reuse or
		//  otherwise tracking scores/intervals without a RefCounted construct.
		Ref<FuzzyTokenMatch> best_match = nullptr;

		while (true) {
			if (!try_match_token(match, token, adjusted_target, offset, p_result->miss_budget)) {
				break;
			}
			if (p_result->can_add_token_match(match)) {
				p_result->score_token_match(match);
				if (best_match.is_null() || best_match->score < match->score) {
					best_match = match;
				}
			}
			if (is_valid_interval(match->interval)) {
				offset = match->interval.x + 1;
			} else {
				break;
			}
			if (match == best_match) {
				match = new_token_match();
			}
		}

		if (best_match.is_null()) {
			return false;
		}

		p_result->add_token_match(best_match);
	}

	return true;
}

void FuzzySearch::set_query(const String &p_query) {
	tokens = p_query.split(" ", false);
	case_sensitive = !p_query.is_lowercase();

	struct TokenComparator {
		bool operator()(const String &A, const String &B) const {
			if (A.length() == B.length()) {
				return A < B;
			}
			return A.length() > B.length();
		}
	};

	// Prioritize matching longer tokens before shorter ones since match overlaps are not accepted
	tokens.sort_custom<TokenComparator>();
}

Ref<FuzzySearchResult> FuzzySearch::search(const String &p_target) {
	if (p_target.is_empty()) {
		return nullptr;
	}

	Ref<FuzzySearchResult> result = new_search_result();

	if (tokens.is_empty()) {
		reset_result(result, p_target);
		return result;
	}

	return fuzzy_search(result, p_target) ? result : nullptr;
}

Vector<Ref<FuzzySearchResult>> FuzzySearch::search_all(const PackedStringArray &p_targets) {
	Vector<Ref<FuzzySearchResult>> results;

	if (p_targets.is_empty()) {
		return results;
	}

	// Just spit out the results list if no query is given.
	if (tokens.is_empty()) {
		for (int i = 0; (i < max_results) && (i < p_targets.size()); i++) {
			Ref<FuzzySearchResult> result = new_search_result();
			reset_result(result, p_targets[0]);
			results.push_back(result);
		}

		return results;
	}

	Ref<FuzzySearchResult> result = new_search_result();

	for (const String &target : p_targets) {
		if (fuzzy_search(result, target)) {
			results.append(result);
			result = new_search_result();
		}
	}

	return sort_and_filter(results);
}

Vector<Ref<FuzzySearchResult>> FuzzySearch::search_all(const String &p_query, const PackedStringArray &p_targets) {
	Ref<FuzzySearch> searcher;
	searcher.instantiate();
	searcher->set_query(p_query);
	return searcher->search_all(p_targets);
}
