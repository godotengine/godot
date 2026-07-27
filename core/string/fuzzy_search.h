/**************************************************************************/
/*  fuzzy_search.h                                                        */
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

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"

class FuzzyTokenMatch;

struct FuzzySearchToken {
	int idx = -1;
	String string;

	bool try_exact_match(FuzzyTokenMatch &p_match, const String &p_target, int p_offset) const;
	bool try_fuzzy_match(FuzzyTokenMatch &p_match, const String &p_target, int p_offset, int p_miss_budget) const;
};

class FuzzyTokenMatch {
	friend struct FuzzySearchToken;
	friend class FuzzySearchMatch;
	friend class FuzzySearch;

	int matched_length = 0;
	int token_length = 0;
	int token_idx = -1;
	Vector2i interval = Vector2i(-1, -1); // x and y are both inclusive indices.

	void add_substring(int p_substring_start, int p_substring_length);
	bool intersects(const Vector2i &p_other_interval) const;
	bool is_case_insensitive(const String &p_original, const String &p_adjusted) const;
	int get_miss_count() const { return token_length - matched_length; }

public:
	int score = 0;
	Vector<Vector2i> substrings; // x is start index, y is length.
};

class FuzzySearchMatch : public RefCounted {
	GDCLASS(FuzzySearchMatch, RefCounted)

	friend class FuzzySearch;

	String target;
	int score = 0;
	int original_index = -1;
	int dir_index = -1;
	Vector<FuzzyTokenMatch> token_matches;
	int miss_budget = 0;
	Vector2i match_interval = Vector2i(-1, -1);

	bool _can_add_token_match(const FuzzyTokenMatch &p_match) const;
	void _score_token_match(FuzzyTokenMatch &p_match, bool p_case_insensitive) const;
	void _add_token_match(const FuzzyTokenMatch &p_match);
	void _maybe_apply_token_order_score_bonus();

protected:
	static void _bind_methods();

public:
	void set_target(const String &p_target) { target = p_target; }
	String get_target() const { return target; }

	void set_score(int p_score) { score = p_score; }
	int get_score() const { return score; }

	void set_original_index(int p_original_index) { original_index = p_original_index; }
	int get_original_index() const { return original_index; }

	void set_dir_index(int p_dir_index) { dir_index = p_dir_index; }
	int get_dir_index() const { return dir_index; }

	Vector<FuzzyTokenMatch> get_token_matches() { return token_matches; }

	TypedArray<Vector2i> get_matched_substrings() const;
};

class FuzzySearch : public RefCounted {
	GDCLASS(FuzzySearch, RefCounted)

	int start_offset = 0;
	int max_results = 100;
	int max_misses = 2;
	bool exact_token_matches = false;
	bool case_sensitive = false;
	bool filter_low_scores = true;
	float filter_factor = 0.1f;
	float filter_cutoff = 30.0f;

	Vector<FuzzySearchToken> _get_tokens(const String &p_query) const;
	void _sort_and_filter(Vector<Ref<FuzzySearchMatch>> &p_results) const;
	bool _search_tokens(const Vector<FuzzySearchToken> &p_tokens, const String &p_target, Ref<FuzzySearchMatch> &r_result) const;
	TypedArray<FuzzySearchMatch> _search_all_bind(const String &p_query, const PackedStringArray &p_targets) const;

protected:
	static void _bind_methods();

public:
	void set_start_offset(int p_offset) {
		ERR_FAIL_COND(p_offset < 0);
		start_offset = p_offset;
	}
	int get_start_offset() const { return start_offset; }

	void set_max_results(int p_max_results) {
		ERR_FAIL_COND(p_max_results <= 0);
		max_results = p_max_results;
	}
	int get_max_results() const { return max_results; }

	void set_max_misses(int p_max_misses) {
		ERR_FAIL_COND(p_max_misses < 0);
		max_misses = p_max_misses;
	}
	int get_max_misses() const { return max_misses; }

	void set_use_exact_tokens(bool p_use_exact_tokens) { exact_token_matches = p_use_exact_tokens; }
	bool get_use_exact_tokens() const { return exact_token_matches; }

	void set_case_sensitive(bool p_case_sensitive);
	bool get_case_sensitive() const { return case_sensitive; }

	void set_filter_low_scores(bool p_filter_low_scores) { filter_low_scores = p_filter_low_scores; }
	bool get_filter_low_scores() const { return filter_low_scores; }

	void set_filter_factor(float p_filter_factor) {
		ERR_FAIL_COND_MSG(p_filter_factor < 0.0f || p_filter_factor > 1.0f, "filter_factor should be in the range [0, 1]");
		filter_factor = p_filter_factor;
	}
	float get_filter_factor() const { return filter_factor; }

	void set_filter_cutoff(float p_filter_cutoff) { filter_cutoff = p_filter_cutoff; }
	float get_filter_cutoff() const { return filter_cutoff; }

	Ref<FuzzySearchMatch> search(const String &p_query, const String &p_target) const;
	Vector<Ref<FuzzySearchMatch>> search_all(const String &p_query, const PackedStringArray &p_targets) const;
};
