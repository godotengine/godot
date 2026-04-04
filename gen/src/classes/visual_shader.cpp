/**************************************************************************/
/*  visual_shader.cpp                                                     */
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

#include <godot_cpp/classes/visual_shader.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/visual_shader_node.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void VisualShader::set_mode(Shader::Mode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("set_mode")._native_ptr(), 3978014962);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

void VisualShader::add_node(VisualShader::Type p_type, const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("add_node")._native_ptr(), 1560769431);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, (p_node != nullptr ? &p_node->_owner : nullptr), &p_position, &p_id_encoded);
}

Ref<VisualShaderNode> VisualShader::get_node(VisualShader::Type p_type, int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_node")._native_ptr(), 3784670312);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<VisualShaderNode>()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return Ref<VisualShaderNode>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<VisualShaderNode>(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded));
}

void VisualShader::set_node_position(VisualShader::Type p_type, int32_t p_id, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("set_node_position")._native_ptr(), 2726660721);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded, &p_position);
}

Vector2 VisualShader::get_node_position(VisualShader::Type p_type, int32_t p_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_node_position")._native_ptr(), 2175036082);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded);
}

PackedInt32Array VisualShader::get_node_list(VisualShader::Type p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_node_list")._native_ptr(), 2370592410);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_type_encoded);
}

int32_t VisualShader::get_valid_node_id(VisualShader::Type p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_valid_node_id")._native_ptr(), 629467342);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type_encoded);
}

void VisualShader::remove_node(VisualShader::Type p_type, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("remove_node")._native_ptr(), 844050912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded);
}

void VisualShader::replace_node(VisualShader::Type p_type, int32_t p_id, const StringName &p_new_class) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("replace_node")._native_ptr(), 3144735253);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded, &p_new_class);
}

bool VisualShader::is_node_connection(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("is_node_connection")._native_ptr(), 3922381898);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_from_node_encoded;
	PtrToArg<int64_t>::encode(p_from_node, &p_from_node_encoded);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_node_encoded;
	PtrToArg<int64_t>::encode(p_to_node, &p_to_node_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_type_encoded, &p_from_node_encoded, &p_from_port_encoded, &p_to_node_encoded, &p_to_port_encoded);
}

bool VisualShader::can_connect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("can_connect_nodes")._native_ptr(), 3922381898);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_from_node_encoded;
	PtrToArg<int64_t>::encode(p_from_node, &p_from_node_encoded);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_node_encoded;
	PtrToArg<int64_t>::encode(p_to_node, &p_to_node_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_type_encoded, &p_from_node_encoded, &p_from_port_encoded, &p_to_node_encoded, &p_to_port_encoded);
}

Error VisualShader::connect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("connect_nodes")._native_ptr(), 3081049573);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_from_node_encoded;
	PtrToArg<int64_t>::encode(p_from_node, &p_from_node_encoded);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_node_encoded;
	PtrToArg<int64_t>::encode(p_to_node, &p_to_node_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type_encoded, &p_from_node_encoded, &p_from_port_encoded, &p_to_node_encoded, &p_to_port_encoded);
}

void VisualShader::disconnect_nodes(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("disconnect_nodes")._native_ptr(), 2268060358);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_from_node_encoded;
	PtrToArg<int64_t>::encode(p_from_node, &p_from_node_encoded);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_node_encoded;
	PtrToArg<int64_t>::encode(p_to_node, &p_to_node_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_from_node_encoded, &p_from_port_encoded, &p_to_node_encoded, &p_to_port_encoded);
}

void VisualShader::connect_nodes_forced(VisualShader::Type p_type, int32_t p_from_node, int32_t p_from_port, int32_t p_to_node, int32_t p_to_port) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("connect_nodes_forced")._native_ptr(), 2268060358);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_from_node_encoded;
	PtrToArg<int64_t>::encode(p_from_node, &p_from_node_encoded);
	int64_t p_from_port_encoded;
	PtrToArg<int64_t>::encode(p_from_port, &p_from_port_encoded);
	int64_t p_to_node_encoded;
	PtrToArg<int64_t>::encode(p_to_node, &p_to_node_encoded);
	int64_t p_to_port_encoded;
	PtrToArg<int64_t>::encode(p_to_port, &p_to_port_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_from_node_encoded, &p_from_port_encoded, &p_to_node_encoded, &p_to_port_encoded);
}

TypedArray<Dictionary> VisualShader::get_node_connections(VisualShader::Type p_type) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_node_connections")._native_ptr(), 1441964831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Dictionary>()));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	return ::godot::internal::_call_native_mb_ret<TypedArray<Dictionary>>(_gde_method_bind, _owner, &p_type_encoded);
}

void VisualShader::attach_node_to_frame(VisualShader::Type p_type, int32_t p_id, int32_t p_frame) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("attach_node_to_frame")._native_ptr(), 2479945279);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	int64_t p_frame_encoded;
	PtrToArg<int64_t>::encode(p_frame, &p_frame_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded, &p_frame_encoded);
}

void VisualShader::detach_node_from_frame(VisualShader::Type p_type, int32_t p_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("detach_node_from_frame")._native_ptr(), 844050912);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_id_encoded;
	PtrToArg<int64_t>::encode(p_id, &p_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_id_encoded);
}

void VisualShader::add_varying(const String &p_name, VisualShader::VaryingMode p_mode, VisualShader::VaryingType p_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("add_varying")._native_ptr(), 2084110726);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_mode_encoded, &p_type_encoded);
}

void VisualShader::remove_varying(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("remove_varying")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

bool VisualShader::has_varying(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("has_varying")._native_ptr(), 3927539163);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void VisualShader::set_graph_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("set_graph_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 VisualShader::get_graph_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(VisualShader::get_class_static()._native_ptr(), StringName("get_graph_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

} // namespace godot
