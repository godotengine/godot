/**************************************************************************/
/*  visual_shader_node_vector_func.hpp                                    */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/visual_shader_node_vector_base.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNodeVectorFunc : public VisualShaderNodeVectorBase {
	GDEXTENSION_CLASS(VisualShaderNodeVectorFunc, VisualShaderNodeVectorBase)

public:
	enum Function {
		FUNC_NORMALIZE = 0,
		FUNC_SATURATE = 1,
		FUNC_NEGATE = 2,
		FUNC_RECIPROCAL = 3,
		FUNC_ABS = 4,
		FUNC_ACOS = 5,
		FUNC_ACOSH = 6,
		FUNC_ASIN = 7,
		FUNC_ASINH = 8,
		FUNC_ATAN = 9,
		FUNC_ATANH = 10,
		FUNC_CEIL = 11,
		FUNC_COS = 12,
		FUNC_COSH = 13,
		FUNC_DEGREES = 14,
		FUNC_EXP = 15,
		FUNC_EXP2 = 16,
		FUNC_FLOOR = 17,
		FUNC_FRACT = 18,
		FUNC_INVERSE_SQRT = 19,
		FUNC_LOG = 20,
		FUNC_LOG2 = 21,
		FUNC_RADIANS = 22,
		FUNC_ROUND = 23,
		FUNC_ROUNDEVEN = 24,
		FUNC_SIGN = 25,
		FUNC_SIN = 26,
		FUNC_SINH = 27,
		FUNC_SQRT = 28,
		FUNC_TAN = 29,
		FUNC_TANH = 30,
		FUNC_TRUNC = 31,
		FUNC_ONEMINUS = 32,
		FUNC_MAX = 33,
	};

	void set_function(VisualShaderNodeVectorFunc::Function p_func);
	VisualShaderNodeVectorFunc::Function get_function() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNodeVectorBase::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeVectorFunc::Function);

