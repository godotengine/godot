/**************************************************************************/
/*  visual_shader_node_compare.hpp                                        */
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
#include <godot_cpp/classes/visual_shader_node.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VisualShaderNodeCompare : public VisualShaderNode {
	GDEXTENSION_CLASS(VisualShaderNodeCompare, VisualShaderNode)

public:
	enum ComparisonType {
		CTYPE_SCALAR = 0,
		CTYPE_SCALAR_INT = 1,
		CTYPE_SCALAR_UINT = 2,
		CTYPE_VECTOR_2D = 3,
		CTYPE_VECTOR_3D = 4,
		CTYPE_VECTOR_4D = 5,
		CTYPE_BOOLEAN = 6,
		CTYPE_TRANSFORM = 7,
		CTYPE_MAX = 8,
	};

	enum Function {
		FUNC_EQUAL = 0,
		FUNC_NOT_EQUAL = 1,
		FUNC_GREATER_THAN = 2,
		FUNC_GREATER_THAN_EQUAL = 3,
		FUNC_LESS_THAN = 4,
		FUNC_LESS_THAN_EQUAL = 5,
		FUNC_MAX = 6,
	};

	enum Condition {
		COND_ALL = 0,
		COND_ANY = 1,
		COND_MAX = 2,
	};

	void set_comparison_type(VisualShaderNodeCompare::ComparisonType p_type);
	VisualShaderNodeCompare::ComparisonType get_comparison_type() const;
	void set_function(VisualShaderNodeCompare::Function p_func);
	VisualShaderNodeCompare::Function get_function() const;
	void set_condition(VisualShaderNodeCompare::Condition p_condition);
	VisualShaderNodeCompare::Condition get_condition() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeCompare::ComparisonType);
VARIANT_ENUM_CAST(VisualShaderNodeCompare::Function);
VARIANT_ENUM_CAST(VisualShaderNodeCompare::Condition);

