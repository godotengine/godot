/**************************************************************************/
/*  visual_shader_node_derivative_func.hpp                                */
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

class VisualShaderNodeDerivativeFunc : public VisualShaderNode {
	GDEXTENSION_CLASS(VisualShaderNodeDerivativeFunc, VisualShaderNode)

public:
	enum OpType {
		OP_TYPE_SCALAR = 0,
		OP_TYPE_VECTOR_2D = 1,
		OP_TYPE_VECTOR_3D = 2,
		OP_TYPE_VECTOR_4D = 3,
		OP_TYPE_MAX = 4,
	};

	enum Function {
		FUNC_SUM = 0,
		FUNC_X = 1,
		FUNC_Y = 2,
		FUNC_MAX = 3,
	};

	enum Precision {
		PRECISION_NONE = 0,
		PRECISION_COARSE = 1,
		PRECISION_FINE = 2,
		PRECISION_MAX = 3,
	};

	void set_op_type(VisualShaderNodeDerivativeFunc::OpType p_type);
	VisualShaderNodeDerivativeFunc::OpType get_op_type() const;
	void set_function(VisualShaderNodeDerivativeFunc::Function p_func);
	VisualShaderNodeDerivativeFunc::Function get_function() const;
	void set_precision(VisualShaderNodeDerivativeFunc::Precision p_precision);
	VisualShaderNodeDerivativeFunc::Precision get_precision() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		VisualShaderNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(VisualShaderNodeDerivativeFunc::OpType);
VARIANT_ENUM_CAST(VisualShaderNodeDerivativeFunc::Function);
VARIANT_ENUM_CAST(VisualShaderNodeDerivativeFunc::Precision);

