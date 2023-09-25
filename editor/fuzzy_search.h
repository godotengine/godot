/**************************************************************************/
/*  fuzzy_search.h                                                   */
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

#ifndef FUZZY_SEARCH_H
#define FUZZY_SEARCH_H

#include "core/templates/rb_set.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"

struct FuzzySearchResult {
	String target;
	int score{};
	RBSet<int> matches;
};

class FuzzySearch {
	int m_total_score{};
	PackedStringArray m_query_tokens;
	Vector<FuzzySearchResult> m_results;

	static FuzzySearchResult fuzzy_search(const String &p_query, const String &p_target, int p_position_offset = 0);

	static FuzzySearchResult fuzzy_search_path_components(const String &p_query_token, const PackedStringArray &p_path_components);

public:
	static String decorate(const FuzzySearchResult &p_result);

	void set_query(const String &p_queue);

	void fuzzy_search_path(const String &p_path);

	const Vector<FuzzySearchResult> &commit();
};

#endif // FUZZY_SEARCH_H
