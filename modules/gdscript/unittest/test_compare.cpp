/*************************************************************************/
/*  test_compare.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "test_compare.h"

bool TestCompare::deep_equal(const Variant &p_left, const Variant &p_right) {
	if (p_left.get_type() == p_right.get_type()) {
		if (p_left.get_type() == Variant::ARRAY && p_right.get_type() == Variant::ARRAY) {
			Array left = p_left;
			Array right = p_right;
			if (left.size() == right.size()) {
				for (int i = 0; i < left.size(); i++) {
					if (!deep_equal(left[i], right[i])) {
						return false;
					}
				}
				return true;
			}
			return false;
		} else if (p_left.get_type() == Variant::DICTIONARY && p_right.get_type() == Variant::DICTIONARY) {
			Dictionary left = p_left;
			Dictionary right = p_right;
			if (left.size() == right.size()) {
				for (int i = 0; i < left.size(); i++) {
					const Variant &key = left.get_key_at_index(i);
					if (!right.has(key)) {
						return false;
					}
					if (!deep_equal(left[key], right[key])) {
						return false;
					}
				}
				return true;
			}
			return false;
		}
	}
	return p_left == p_right;
}
