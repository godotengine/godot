/**************************************************************************/
/*  visual_shader_node_derivative_func.cpp                                */
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

#include <godot_cpp/classes/visual_shader_node_derivative_func.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VisualShaderNodeDerivativeFunc::set_op_type(VisualShaderNodeDerivativeFunc::OpType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("set_op_type")._native_ptr(), 377800221);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

VisualShaderNodeDerivativeFunc::OpType VisualShaderNodeDerivativeFunc::get_op_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("get_op_type")._native_ptr(), 3997800514);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeDerivativeFunc::OpType(0)));
	return (VisualShaderNodeDerivativeFunc::OpType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeDerivativeFunc::set_function(VisualShaderNodeDerivativeFunc::Function p_func) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("set_function")._native_ptr(), 1944704156);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_func_encoded;
	PtrToArg<int64_t>::encode(p_func, &p_func_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_func_encoded);
}

VisualShaderNodeDerivativeFunc::Function VisualShaderNodeDerivativeFunc::get_function() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("get_function")._native_ptr(), 2389093396);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeDerivativeFunc::Function(0)));
	return (VisualShaderNodeDerivativeFunc::Function)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeDerivativeFunc::set_precision(VisualShaderNodeDerivativeFunc::Precision p_precision) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("set_precision")._native_ptr(), 797270566);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_precision_encoded;
	PtrToArg<int64_t>::encode(p_precision, &p_precision_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_precision_encoded);
}

VisualShaderNodeDerivativeFunc::Precision VisualShaderNodeDerivativeFunc::get_precision() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeDerivativeFunc::get_class_static()._native_ptr(), StringName("get_precision")._native_ptr(), 3822547323);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeDerivativeFunc::Precision(0)));
	return (VisualShaderNodeDerivativeFunc::Precision)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
