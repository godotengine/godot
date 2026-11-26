/**************************************************************************/
/*  gdscript_constants.cpp                                                */
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

#include "gdscript_constants.h"

#include "core/math/math_defs.h"

static constexpr int _gdscript_constant_count = 4;
static constexpr const char *_gdscript_constant_names[] = { "PI", "TAU", "INF", "NAN" };
static constexpr const double _gdscript_constant_values[] = { Math::PI, Math::TAU, Math::INF, Math::NaN };

int GDScriptConstants::get_constant_count() {
	return _gdscript_constant_count;
}

const char *GDScriptConstants::get_constant_name(int p_idx) {
	return _gdscript_constant_names[p_idx];
}

Variant GDScriptConstants::get_constant_value(int p_idx) {
	return Variant(_gdscript_constant_values[p_idx]);
}

bool GDScriptConstants::is_constant(const StringName &p_name) {
	return get_constant_index(p_name) >= 0;
}

int GDScriptConstants::get_constant_index(const StringName &p_name) {
	for (int i = 0; i < _gdscript_constant_count; i++) {
		if (_gdscript_constant_names[i] == p_name) {
			return i;
		}
	}
	return -1;
}
