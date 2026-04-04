/**************************************************************************/
/*  visual_shader_node_cubemap.cpp                                        */
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

#include <godot_cpp/classes/visual_shader_node_cubemap.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture_layered.hpp>

namespace godot {

void VisualShaderNodeCubemap::set_source(VisualShaderNodeCubemap::Source p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("set_source")._native_ptr(), 1625400621);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

VisualShaderNodeCubemap::Source VisualShaderNodeCubemap::get_source() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("get_source")._native_ptr(), 2222048781);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeCubemap::Source(0)));
	return (VisualShaderNodeCubemap::Source)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeCubemap::set_cube_map(const Ref<TextureLayered> &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("set_cube_map")._native_ptr(), 1278366092);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_value != nullptr ? &p_value->_owner : nullptr));
}

Ref<TextureLayered> VisualShaderNodeCubemap::get_cube_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("get_cube_map")._native_ptr(), 3984243839);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<TextureLayered>()));
	return Ref<TextureLayered>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<TextureLayered>(_gde_method_bind, _owner));
}

void VisualShaderNodeCubemap::set_texture_type(VisualShaderNodeCubemap::TextureType p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("set_texture_type")._native_ptr(), 1899718876);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_value_encoded;
	PtrToArg<int64_t>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_value_encoded);
}

VisualShaderNodeCubemap::TextureType VisualShaderNodeCubemap::get_texture_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeCubemap::get_class_static()._native_ptr(), StringName("get_texture_type")._native_ptr(), 3356498888);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeCubemap::TextureType(0)));
	return (VisualShaderNodeCubemap::TextureType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
