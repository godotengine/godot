/**************************************************************************/
/*  visual_shader_node_texture_parameter.cpp                              */
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

#include <godot_cpp/classes/visual_shader_node_texture_parameter.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void VisualShaderNodeTextureParameter::set_texture_type(VisualShaderNodeTextureParameter::TextureType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("set_texture_type")._native_ptr(), 2227296876);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded);
}

VisualShaderNodeTextureParameter::TextureType VisualShaderNodeTextureParameter::get_texture_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("get_texture_type")._native_ptr(), 367922070);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeTextureParameter::TextureType(0)));
	return (VisualShaderNodeTextureParameter::TextureType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeTextureParameter::set_color_default(VisualShaderNodeTextureParameter::ColorDefault p_color) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("set_color_default")._native_ptr(), 4217624432);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_color_encoded;
	PtrToArg<int64_t>::encode(p_color, &p_color_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_color_encoded);
}

VisualShaderNodeTextureParameter::ColorDefault VisualShaderNodeTextureParameter::get_color_default() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("get_color_default")._native_ptr(), 3837060134);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeTextureParameter::ColorDefault(0)));
	return (VisualShaderNodeTextureParameter::ColorDefault)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeTextureParameter::set_texture_filter(VisualShaderNodeTextureParameter::TextureFilter p_filter) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("set_texture_filter")._native_ptr(), 2147684752);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_encoded);
}

VisualShaderNodeTextureParameter::TextureFilter VisualShaderNodeTextureParameter::get_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("get_texture_filter")._native_ptr(), 4184490817);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeTextureParameter::TextureFilter(0)));
	return (VisualShaderNodeTextureParameter::TextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeTextureParameter::set_texture_repeat(VisualShaderNodeTextureParameter::TextureRepeat p_repeat) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("set_texture_repeat")._native_ptr(), 2036143070);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_repeat_encoded;
	PtrToArg<int64_t>::encode(p_repeat, &p_repeat_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_repeat_encoded);
}

VisualShaderNodeTextureParameter::TextureRepeat VisualShaderNodeTextureParameter::get_texture_repeat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("get_texture_repeat")._native_ptr(), 1690132794);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeTextureParameter::TextureRepeat(0)));
	return (VisualShaderNodeTextureParameter::TextureRepeat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void VisualShaderNodeTextureParameter::set_texture_source(VisualShaderNodeTextureParameter::TextureSource p_source) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("set_texture_source")._native_ptr(), 1212687372);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_source_encoded;
	PtrToArg<int64_t>::encode(p_source, &p_source_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_source_encoded);
}

VisualShaderNodeTextureParameter::TextureSource VisualShaderNodeTextureParameter::get_texture_source() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShaderNodeTextureParameter::get_class_static()._native_ptr(), StringName("get_texture_source")._native_ptr(), 2039092262);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (VisualShaderNodeTextureParameter::TextureSource(0)));
	return (VisualShaderNodeTextureParameter::TextureSource)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
