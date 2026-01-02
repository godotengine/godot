/**************************************************************************/
/*  plural_rules.h                                                        */
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
#include "core/templates/lru.h"

class Expression;

class PluralRules : public Object {
	GDSOFTCLASS(PluralRules, Object);

	mutable LRUCache<int, int> cache;

	// These two fields are initialized in the constructor.
	const int nplurals;
	const String plural;

	// Cache temporary variables related to `evaluate()` to make it faster.
	class EQNode : public RefCounted {
		GDSOFTCLASS(EQNode, RefCounted);

	public:
		String regex;
		Ref<EQNode> left;
		Ref<EQNode> right;
	};
	Ref<EQNode> equi_tests;
	Ref<Expression> expr;

	int _find_unquoted(const String &p_src, char32_t p_chr) const;
	int _eq_test(const Array &p_input_val, const Ref<EQNode> &p_node, const Variant &p_result) const;
	void _cache_plural_tests(const String &p_plural_rule, Ref<EQNode> &p_node);

	PluralRules(int p_nplurals, const String &p_plural);

public:
	int evaluate(int p_n) const;

	int get_nplurals() const { return nplurals; }
	String get_plural() const { return plural; }

	static PluralRules *parse(const String &p_rules);
};
