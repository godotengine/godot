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

#include "core/variant/typed_array.h"

const String boundary_chars = "/\\-_. ";

static bool _is_valid_interval(const Vector2i &p_interval) {
	// Empty intervals are represented as (-1, -1).
	return p_interval.x >= 0 && p_interval.y >= p_interval.x;
}

static Vector2i _extend_interval(const Vector2i &p_a, const Vector2i &p_b) {
	if (!_is_valid_interval(p_a)) {
		return p_b;
	}
	if (!_is_valid_interval(p_b)) {
		return p_a;
	}
	return Vector2i(MIN(p_a.x, p_b.x), MAX(p_a.y, p_b.y));
}

static bool _is_word_boundary(const String &p_str, int p_index) {
	if (p_index == -1 || p_index == p_str.size()) {
		return true;
	}
	return boundary_chars.find_char(p_str[p_index]) != -1;
}

bool FuzzySearchToken::try_exact_match(FuzzyTokenMatch &p_match, const String &p_target, int p_offset) const {
	p_match.token_idx = idx;
	p_match.token_length = string.length();
	int match_idx = p_target.find(string, p_offset);
	if (match_idx == -1) {
		return false;
	}
	p_match.add_substring(match_idx, string.length());
	return true;
}

bool FuzzySearchToken::try_fuzzy_match(FuzzyTokenMatch &p_match, const String &p_target, int p_offset, int p_miss_budget) const {
	p_match.token_idx = idx;
	p_match.token_length = string.length();
	int run_start = -1;
	int run_len = 0;

	// Search for the subsequence p_token in p_target starting from p_offset, recording each substring for
	// later scoring and display.
	for (int i = 0; i < string.length(); i++) {
		int new_offset = p_target.find_char(string[i], p_offset);
		if (new_offset < 0) {
			p_miss_budget--;
			if (p_miss_budget < 0) {
				return false;
			}
		} else {
			if (run_start == -1 || p_offset != new_offset) {
				if (run_start != -1) {
					p_match.add_substring(run_start, run_len);
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
		p_match.add_substring(run_start, run_len);
	}

	return true;
}

void FuzzyTokenMatch::add_substring(int p_substring_start, int p_substring_length) {
	substrings.append(Vector2i(p_substring_start, p_substring_length));
	matched_length += p_substring_length;
	Vector2i substring_interval = { p_substring_start, p_substring_start + p_substring_length - 1 };
	interval = _extend_interval(interval, substring_interval);
}

bool FuzzyTokenMatch::intersects(const Vector2i &p_other_interval) const {
	if (!_is_valid_interval(interval) || !_is_valid_interval(p_other_interval)) {
		return false;
	}
	return interval.y >= p_other_interval.x && interval.x <= p_other_interval.y;
}

bool FuzzySearchResult::can_add_token_match(const FuzzyTokenMatch &p_match) const {
	if (p_match.get_miss_count() > miss_budget) {
		return false;
	}

	if (p_match.intersects(match_interval)) {
		if (token_matches.size() == 1) {
			return false;
		}
		for (const FuzzyTokenMatch &existing_match : token_matches) {
			if (existing_match.intersects(p_match.interval)) {
				return false;
			}
		}
	}

	return true;
}

bool FuzzyTokenMatch::is_case_insensitive(const String &p_original, const String &p_adjusted) const {
	for (const Vector2i &substr : substrings) {
		const int end = substr.x + substr.y;
		for (int i = substr.x; i < end; i++) {
			if (p_original[i] != p_adjusted[i]) {
				return true;
			}
		}
	}
	return false;
}

void FuzzySearchResult::score_token_match(FuzzyTokenMatch &p_match, bool p_case_insensitive) const {
	// This can always be tweaked more. The intuition is that exact matches should almost always
	// be prioritized over broken up matches, and other criteria more or less act as tie breakers.

	p_match.score = -20 * p_match.get_miss_count() - (p_case_insensitive ? 3 : 0);

	for (const Vector2i &substring : p_match.substrings) {
		// Score longer substrings higher than short substrings.
		int substring_score = substring.y * substring.y;
		// Score matches deeper in path higher than shallower matches
		if (substring.x > dir_index) {
			substring_score *= 2;
		}
		// Score matches on a word boundary higher than matches within a word
		if (_is_word_boundary(target, substring.x - 1) || _is_word_boundary(target, substring.x + substring.y)) {
			substring_score += 4;
		}
		// Score exact query matches higher than non-compact subsequence matches
		if (substring.y == p_match.token_length) {
			substring_score += 100;
		}
		p_match.score += substring_score;
	}
}

void FuzzySearchResult::maybe_apply_score_bonus() {
	// This adds a small bonus to results which match tokens in the same order they appear in the query.
	if (token_matches.is_empty()) {
		return;
	}

	int *token_range_starts = (int *)alloca(sizeof(int) * token_matches.size());

	for (const FuzzyTokenMatch &match : token_matches) {
		token_range_starts[match.token_idx] = match.interval.x;
	}

	int last = token_range_starts[0];
	for (int i = 1; i < token_matches.size(); i++) {
		if (last > token_range_starts[i]) {
			return;
		}
		last = token_range_starts[i];
	}

	score += 1;
}

void FuzzySearchResult::add_token_match(const FuzzyTokenMatch &p_match) {
	score += p_match.score;
	match_interval = _extend_interval(match_interval, p_match.interval);
	miss_budget -= p_match.get_miss_count();
	token_matches.append(p_match);
}

void FuzzyMatch::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_target", "target"), &FuzzyMatch::set_target);
	ClassDB::bind_method(D_METHOD("get_target"), &FuzzyMatch::get_target);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "target"), "set_target", "get_target");

	ClassDB::bind_method(D_METHOD("set_score", "score"), &FuzzyMatch::set_score);
	ClassDB::bind_method(D_METHOD("get_score"), &FuzzyMatch::get_score);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "score"), "set_score", "get_score");

	ClassDB::bind_method(D_METHOD("set_original_index", "original_index"), &FuzzyMatch::set_original_index);
	ClassDB::bind_method(D_METHOD("get_original_index"), &FuzzyMatch::get_original_index);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "original_index"), "set_original_index", "get_original_index");

	ClassDB::bind_method(D_METHOD("get_matched_substrings"), &FuzzyMatch::get_matched_substrings);
}

TypedArray<Vector2i> FuzzyMatch::get_matched_substrings() {
	TypedArray<Vector2i> substrings;
	for (const FuzzyTokenMatch &match : result.token_matches) {
		for (const Vector2i &substring : match.substrings) {
			substrings.append(substring);
		}
	}
	return substrings;
}

void remove_low_scores(Vector<FuzzySearchResult> &p_results, float p_cull_score) {
	// Removes all results with score < p_cull_score in-place.
	int i = 0;
	int j = p_results.size() - 1;
	FuzzySearchResult *results = p_results.ptrw();

	while (true) {
		// Advances i to an element to remove and j to an element to keep.
		while (j >= i && results[j].score < p_cull_score) {
			j--;
		}
		while (i < j && results[i].score >= p_cull_score) {
			i++;
		}
		if (i >= j) {
			break;
		}
		results[i++] = results[j--];
	}

	p_results.resize(j + 1);
}

void FuzzySearch::update_tokens() {
	tokens.clear();

	for (const String &string : query.split(" ", false)) {
		tokens.append({
				static_cast<int>(tokens.size()),
				case_sensitive ? string : string.to_lower(),
		});
	}

	struct TokenComparator {
		bool operator()(const FuzzySearchToken &A, const FuzzySearchToken &B) const {
			if (A.string.length() == B.string.length()) {
				return A.idx < B.idx;
			}
			return A.string.length() > B.string.length();
		}
	};

	// Prioritize matching longer tokens before shorter ones since match overlaps are not accepted.
	tokens.sort_custom<TokenComparator>();
	dirty_tokens = false;
}

void FuzzySearch::sort_and_filter(Vector<FuzzySearchResult> &p_results) const {
	if (p_results.is_empty()) {
		return;
	}

	if (filter_low_scores) {
		float avg_score = 0;
		float max_score = 0;

		for (const FuzzySearchResult &result : p_results) {
			avg_score += result.score;
			max_score = MAX(max_score, result.score);
		}

		// TODO: Tune scoring and culling here to display fewer subsequence soup matches when good matches
		//  are available.
		avg_score /= p_results.size();
		float cull_score = MIN(filter_cutoff, Math::lerp(avg_score, max_score, filter_factor));
		remove_low_scores(p_results, cull_score);
	}

	struct FuzzySearchResultComparator {
		bool operator()(const FuzzySearchResult &p_lhs, const FuzzySearchResult &p_rhs) const {
			// Sort on (score, length, alphanumeric) to ensure consistent ordering.
			if (p_lhs.score == p_rhs.score) {
				if (p_lhs.target.length() == p_rhs.target.length()) {
					return p_lhs.target < p_rhs.target;
				}
				return p_lhs.target.length() < p_rhs.target.length();
			}
			return p_lhs.score > p_rhs.score;
		}
	};

	SortArray<FuzzySearchResult, FuzzySearchResultComparator> sorter;

	if (p_results.size() > max_results) {
		sorter.partial_sort(0, p_results.size(), max_results, p_results.ptrw());
		p_results.resize(max_results);
	} else {
		sorter.sort(p_results.ptrw(), p_results.size());
	}
}

void FuzzySearch::set_query(const String &p_query) {
	query = p_query;
	dirty_tokens = true;
}

void FuzzySearch::set_case_sensitive(bool p_case_sensitive) {
	case_sensitive = p_case_sensitive;
	dirty_tokens = true;
}

bool FuzzySearch::search(const String &p_target, FuzzySearchResult &r_result) const {
	if (dirty_tokens) {
		((FuzzySearch *)this)->update_tokens();
	}

	r_result.target = p_target;
	r_result.dir_index = p_target.rfind_char('/');
	r_result.miss_budget = max_misses;

	String adjusted_target = case_sensitive ? p_target : p_target.to_lower();

	// For each token, eagerly generate subsequences starting from index 0 and keep the best scoring one
	// which does not conflict with prior token matches. This is not ensured to find the highest scoring
	// combination of matches, or necessarily the highest scoring single subsequence, as it only considers
	// eager subsequences for a given index, and likewise eagerly finds matches for each token in sequence.
	for (const FuzzySearchToken &token : tokens) {
		FuzzyTokenMatch best_match;
		int offset = start_offset;

		while (true) {
			FuzzyTokenMatch match;
			if (exact_token_matches) {
				if (!token.try_exact_match(match, adjusted_target, offset)) {
					break;
				}
			} else {
				if (!token.try_fuzzy_match(match, adjusted_target, offset, r_result.miss_budget)) {
					break;
				}
			}
			if (r_result.can_add_token_match(match)) {
				r_result.score_token_match(match, match.is_case_insensitive(p_target, adjusted_target));
				if (best_match.token_idx == -1 || best_match.score < match.score) {
					best_match = match;
				}
			}
			if (_is_valid_interval(match.interval)) {
				offset = match.interval.x + 1;
			} else {
				break;
			}
		}

		if (best_match.token_idx == -1) {
			return false;
		}

		r_result.add_token_match(best_match);
	}

	if (r_result.match_interval.x == -1) {
		// Reject matches which rely entirely on misses.
		return false;
	}

	r_result.maybe_apply_score_bonus();
	return true;
}

void FuzzySearch::search_all(const PackedStringArray &p_targets, Vector<FuzzySearchResult> &r_results) const {
	r_results.clear();

	for (int i = 0; i < p_targets.size(); i++) {
		FuzzySearchResult result;
		result.original_index = i;
		if (search(p_targets[i], result)) {
			r_results.append(result);
		}
	}

	sort_and_filter(r_results);
}

Ref<FuzzyMatch> FuzzySearch::search_wrapped(const String &p_target) const {
	FuzzySearchResult result;
	if (search(p_target, result)) {
		Ref<FuzzyMatch> wrapped;
		wrapped.instantiate();
		wrapped->result = result;
		return wrapped;
	}
	return nullptr;
}

TypedArray<Ref<FuzzyMatch>> FuzzySearch::search_all_wrapped(const PackedStringArray &p_targets) const {
	Vector<FuzzySearchResult> results;
	TypedArray<Ref<FuzzyMatch>> wrapped_results;

	search_all(p_targets, results);
	for (FuzzySearchResult result : results) {
		Ref<FuzzyMatch> wrapped;
		wrapped.instantiate();
		wrapped->result = result;
		wrapped_results.append(wrapped);
	}

	return wrapped_results;
}

void FuzzySearch::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_query", "query"), &FuzzySearch::set_query);
	ClassDB::bind_method(D_METHOD("get_query"), &FuzzySearch::get_query);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "query"), "set_query", "get_query");

	ClassDB::bind_method(D_METHOD("set_max_results", "max_results"), &FuzzySearch::set_max_results);
	ClassDB::bind_method(D_METHOD("get_max_results"), &FuzzySearch::get_max_results);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_results"), "set_max_results", "get_max_results");

	ClassDB::bind_method(D_METHOD("set_max_misses", "max_misses"), &FuzzySearch::set_max_misses);
	ClassDB::bind_method(D_METHOD("get_max_misses"), &FuzzySearch::get_max_misses);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_misses"), "set_max_misses", "get_max_misses");

	ClassDB::bind_method(D_METHOD("set_use_exact_tokens", "use_exact_tokens"), &FuzzySearch::set_use_exact_tokens);
	ClassDB::bind_method(D_METHOD("get_use_exact_tokens"), &FuzzySearch::get_use_exact_tokens);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_exact_tokens"), "set_use_exact_tokens", "get_use_exact_tokens");

	ClassDB::bind_method(D_METHOD("set_case_sensitive", "case_sensitive"), &FuzzySearch::set_case_sensitive);
	ClassDB::bind_method(D_METHOD("get_case_sensitive"), &FuzzySearch::get_case_sensitive);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "case_sensitive"), "set_case_sensitive", "get_case_sensitive");

	ClassDB::bind_method(D_METHOD("set_filter_low_scores", "filter_low_scores"), &FuzzySearch::set_filter_low_scores);
	ClassDB::bind_method(D_METHOD("get_filter_low_scores"), &FuzzySearch::get_filter_low_scores);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_low_scores"), "set_filter_low_scores", "get_filter_low_scores");

	ClassDB::bind_method(D_METHOD("set_filter_factor", "filter_factor"), &FuzzySearch::set_filter_factor);
	ClassDB::bind_method(D_METHOD("get_filter_factor"), &FuzzySearch::get_filter_factor);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "filter_factor"), "set_filter_factor", "get_filter_factor");

	ClassDB::bind_method(D_METHOD("set_filter_cutoff", "filter_cutoff"), &FuzzySearch::set_filter_cutoff);
	ClassDB::bind_method(D_METHOD("get_filter_cutoff"), &FuzzySearch::get_filter_cutoff);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "filter_cutoff"), "set_filter_cutoff", "get_filter_cutoff");

	ClassDB::bind_method(D_METHOD("search", "target"), &FuzzySearch::search_wrapped);
	ClassDB::bind_method(D_METHOD("search_all", "targets"), &FuzzySearch::search_all_wrapped);
}
