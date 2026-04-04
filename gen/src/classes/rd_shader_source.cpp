/**************************************************************************/
/*  rd_shader_source.cpp                                                  */
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

#include <godot_cpp/classes/rd_shader_source.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void RDShaderSource::set_stage_source(RenderingDevice::ShaderStage p_stage, const String &p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDShaderSource::get_class_static()._native_ptr(), StringName("set_stage_source")._native_ptr(), 620821314);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_stage_encoded;
	PtrToArg<int64_t>::encode(p_stage, &p_stage_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_stage_encoded, &p_source);
}

String RDShaderSource::get_stage_source(RenderingDevice::ShaderStage p_stage) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDShaderSource::get_class_static()._native_ptr(), StringName("get_stage_source")._native_ptr(), 3354920045);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_stage_encoded;
	PtrToArg<int64_t>::encode(p_stage, &p_stage_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_stage_encoded);
}

void RDShaderSource::set_language(RenderingDevice::ShaderLanguage p_language) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDShaderSource::get_class_static()._native_ptr(), StringName("set_language")._native_ptr(), 3422186742);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_language_encoded;
	PtrToArg<int64_t>::encode(p_language, &p_language_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_language_encoded);
}

RenderingDevice::ShaderLanguage RDShaderSource::get_language() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RDShaderSource::get_class_static()._native_ptr(), StringName("get_language")._native_ptr(), 1063538261);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RenderingDevice::ShaderLanguage(0)));
	return (RenderingDevice::ShaderLanguage)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
