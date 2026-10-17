/**************************************************************************/
/*  plural_rules.cpp                                                      */
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

#include "plural_rules.h"

#include "core/math/expression.h"

int PluralRules::_eq_test(const Array &p_input_val, const Ref<EQNode> &p_node, const Variant &p_result) const {
	if (p_node.is_null()) {
		return p_result;
	}

	static const Vector<String> input_name = { "n" };

	Error err = expr->parse(p_node->regex, input_name);
	ERR_FAIL_COND_V_MSG(err != OK, 0, vformat("Cannot parse expression \"%s\". Error: %s", p_node->regex, expr->get_error_text()));

	Variant result = expr->execute(p_input_val);
	ERR_FAIL_COND_V_MSG(expr->has_execute_failed(), 0, vformat("Cannot evaluate expression \"%s\".", p_node->regex));

	if (bool(result)) {
		return _eq_test(p_input_val, p_node->left, result);
	} else {
		return _eq_test(p_input_val, p_node->right, result);
	}
}

int PluralRules::_find_unquoted(const String &p_src, char32_t p_chr) const {
	const int len = p_src.length();
	if (len == 0) {
		return -1;
	}

	const char32_t *src = p_src.get_data();
	bool in_quote = false;
	for (int i = 0; i < len; i++) {
		if (in_quote) {
			if (src[i] == ')') {
				in_quote = false;
			}
		} else {
			if (src[i] == '(') {
				in_quote = true;
			} else if (src[i] == p_chr) {
				return i;
			}
		}
	}

	return -1;
}

void PluralRules::_cache_plural_tests(const String &p_plural_rule, Ref<EQNode> &p_node) {
	// Some examples of p_plural_rule passed in can have the form:
	// "n==0 ? 0 : n==1 ? 1 : n==2 ? 2 : n%100>=3 && n%100<=10 ? 3 : n%100>=11 && n%100<=99 ? 4 : 5" (Arabic)
	// "n >= 2" (French) // When evaluating the last, especially careful with this one.
	// "n != 1" (English)

	String rule = p_plural_rule;
	if (rule.begins_with("(") && rule.ends_with(")")) {
		int bcount = 0;
		for (int i = 1; i < rule.length() - 1 && bcount >= 0; i++) {
			if (rule[i] == '(') {
				bcount++;
			} else if (rule[i] == ')') {
				bcount--;
			}
		}
		if (bcount == 0) {
			rule = rule.substr(1, rule.length() - 2);
		}
	}

	int first_ques_mark = _find_unquoted(rule, '?');
	int first_colon = _find_unquoted(rule, ':');

	if (first_ques_mark == -1) {
		p_node->regex = rule.strip_edges();
		return;
	}

	p_node->regex = rule.substr(0, first_ques_mark).strip_edges();

	p_node->left.instantiate();
	_cache_plural_tests(rule.substr(first_ques_mark + 1, first_colon - first_ques_mark - 1).strip_edges(), p_node->left);
	p_node->right.instantiate();
	_cache_plural_tests(rule.substr(first_colon + 1).strip_edges(), p_node->right);
}

int PluralRules::evaluate(int p_n) const {
	const int *cached = cache.getptr(p_n);
	if (cached) {
		return *cached;
	}

	const Array &input_val = { p_n };
	int index = _eq_test(input_val, equi_tests, 0);
	cache.insert(p_n, index);
	return index;
}

PluralRules::PluralRules(int p_nplurals, const String &p_plural) :
		nplurals(p_nplurals),
		plural(p_plural) {
	equi_tests.instantiate();
	_cache_plural_tests(plural, equi_tests);

	expr.instantiate();
}

PluralRules *PluralRules::parse(const String &p_rules) {
	// `p_rules` should be in the format "nplurals=<N>; plural=<Expression>;".

	const int nplurals_eq = p_rules.find_char('=');
	ERR_FAIL_COND_V_MSG(nplurals_eq == -1, nullptr, "Invalid plural rules format. Missing equal sign for `nplurals`.");

	const int nplurals_semi_col = p_rules.find_char(';', nplurals_eq);
	ERR_FAIL_COND_V_MSG(nplurals_semi_col == -1, nullptr, "Invalid plural rules format. Missing semicolon for `nplurals`.");

	const String nplurals_str = p_rules.substr(nplurals_eq + 1, nplurals_semi_col - (nplurals_eq + 1)).strip_edges();
	ERR_FAIL_COND_V_MSG(!nplurals_str.is_valid_int(), nullptr, "Invalid plural rules format. `nplurals` should be an integer.");

	const int nplurals = nplurals_str.to_int();
	ERR_FAIL_COND_V_MSG(nplurals < 1, nullptr, "Invalid plural rules format. `nplurals` should be at least 1.");

	const int expression_eq = p_rules.find_char('=', nplurals_semi_col + 1);
	ERR_FAIL_COND_V_MSG(expression_eq == -1, nullptr, "Invalid plural rules format. Missing equal sign for `plural`.");

	int expression_end = p_rules.rfind_char(';');
	if (expression_end == -1) {
		WARN_PRINT("Invalid plural rules format. Missing semicolon at the end of `plural` expression. Assuming ends at the end of the string.");
		expression_end = p_rules.length();
	}

	const int expression_start = expression_eq + 1;
	ERR_FAIL_COND_V_MSG(expression_end <= expression_start, nullptr, "Invalid plural rules format. `plural` expression is empty.");

	const String &plural = p_rules.substr(expression_start, expression_end - expression_start).strip_edges();
	return memnew(PluralRules(nplurals, plural));
}
