/**************************************************************************/
/*  shader.cpp                                                            */
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

#include <godot_cpp/classes/shader.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

Shader::Mode Shader::get_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("get_mode")._native_ptr(), 3392948163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Shader::Mode(0)));
	return (Shader::Mode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Shader::set_code(const String &p_code) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("set_code")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_code);
}

String Shader::get_code() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("get_code")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Shader::set_default_texture_parameter(const StringName &p_name, const Ref<Texture> &p_texture, int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("set_default_texture_parameter")._native_ptr(), 3850209648);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_index_encoded);
}

Ref<Texture> Shader::get_default_texture_parameter(const StringName &p_name, int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("get_default_texture_parameter")._native_ptr(), 4213877425);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture>()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return Ref<Texture>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture>(_gde_method_bind, _owner, &p_name, &p_index_encoded));
}

Array Shader::get_shader_uniform_list(bool p_get_groups) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("get_shader_uniform_list")._native_ptr(), 1230511656);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	int8_t p_get_groups_encoded;
	PtrToArg<bool>::encode(p_get_groups, &p_get_groups_encoded);
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner, &p_get_groups_encoded);
}

void Shader::inspect_native_shader_code() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Shader::get_class_static()._native_ptr(), StringName("inspect_native_shader_code")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
