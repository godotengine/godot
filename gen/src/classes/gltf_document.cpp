/**************************************************************************/
/*  gltf_document.cpp                                                     */
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

#include <godot_cpp/classes/gltf_document.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/gltf_document_extension.hpp>
#include <godot_cpp/classes/gltf_object_model_property.hpp>
#include <godot_cpp/classes/gltf_state.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/node_path.hpp>

namespace godot {

void GLTFDocument::set_image_format(const String &p_image_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_image_format")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_image_format);
}

String GLTFDocument::get_image_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_image_format")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFDocument::set_lossy_quality(float p_lossy_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_lossy_quality")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_lossy_quality_encoded;
	PtrToArg<double>::encode(p_lossy_quality, &p_lossy_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_lossy_quality_encoded);
}

float GLTFDocument::get_lossy_quality() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_lossy_quality")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFDocument::set_fallback_image_format(const String &p_fallback_image_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_fallback_image_format")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fallback_image_format);
}

String GLTFDocument::get_fallback_image_format() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_fallback_image_format")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void GLTFDocument::set_fallback_image_quality(float p_fallback_image_quality) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_fallback_image_quality")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fallback_image_quality_encoded;
	PtrToArg<double>::encode(p_fallback_image_quality, &p_fallback_image_quality_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fallback_image_quality_encoded);
}

float GLTFDocument::get_fallback_image_quality() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_fallback_image_quality")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void GLTFDocument::set_root_node_mode(GLTFDocument::RootNodeMode p_root_node_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_root_node_mode")._native_ptr(), 463633402);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_root_node_mode_encoded;
	PtrToArg<int64_t>::encode(p_root_node_mode, &p_root_node_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_root_node_mode_encoded);
}

GLTFDocument::RootNodeMode GLTFDocument::get_root_node_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_root_node_mode")._native_ptr(), 948057992);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GLTFDocument::RootNodeMode(0)));
	return (GLTFDocument::RootNodeMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void GLTFDocument::set_visibility_mode(GLTFDocument::VisibilityMode p_visibility_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("set_visibility_mode")._native_ptr(), 2803579218);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_visibility_mode_encoded;
	PtrToArg<int64_t>::encode(p_visibility_mode, &p_visibility_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_visibility_mode_encoded);
}

GLTFDocument::VisibilityMode GLTFDocument::get_visibility_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_visibility_mode")._native_ptr(), 3885445962);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (GLTFDocument::VisibilityMode(0)));
	return (GLTFDocument::VisibilityMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error GLTFDocument::append_from_file(const String &p_path, const Ref<GLTFState> &p_state, uint32_t p_flags, const String &p_base_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("append_from_file")._native_ptr(), 866380864);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, (p_state != nullptr ? &p_state->_owner : nullptr), &p_flags_encoded, &p_base_path);
}

Error GLTFDocument::append_from_buffer(const PackedByteArray &p_bytes, const String &p_base_path, const Ref<GLTFState> &p_state, uint32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("append_from_buffer")._native_ptr(), 1616081266);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_bytes, &p_base_path, (p_state != nullptr ? &p_state->_owner : nullptr), &p_flags_encoded);
}

Error GLTFDocument::append_from_scene(Node *p_node, const Ref<GLTFState> &p_state, uint32_t p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("append_from_scene")._native_ptr(), 1622574258);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_flags_encoded;
	PtrToArg<int64_t>::encode(p_flags, &p_flags_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_node != nullptr ? &p_node->_owner : nullptr), (p_state != nullptr ? &p_state->_owner : nullptr), &p_flags_encoded);
}

Node *GLTFDocument::generate_scene(const Ref<GLTFState> &p_state, float p_bake_fps, bool p_trimming, bool p_remove_immutable_tracks) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("generate_scene")._native_ptr(), 596118388);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	double p_bake_fps_encoded;
	PtrToArg<double>::encode(p_bake_fps, &p_bake_fps_encoded);
	int8_t p_trimming_encoded;
	PtrToArg<bool>::encode(p_trimming, &p_trimming_encoded);
	int8_t p_remove_immutable_tracks_encoded;
	PtrToArg<bool>::encode(p_remove_immutable_tracks, &p_remove_immutable_tracks_encoded);
	return ::godot::internal::_call_native_mb_ret_obj<Node>(_gde_method_bind, _owner, (p_state != nullptr ? &p_state->_owner : nullptr), &p_bake_fps_encoded, &p_trimming_encoded, &p_remove_immutable_tracks_encoded);
}

PackedByteArray GLTFDocument::generate_buffer(const Ref<GLTFState> &p_state) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("generate_buffer")._native_ptr(), 741783455);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, (p_state != nullptr ? &p_state->_owner : nullptr));
}

Error GLTFDocument::write_to_filesystem(const Ref<GLTFState> &p_state, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("write_to_filesystem")._native_ptr(), 1784551478);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_state != nullptr ? &p_state->_owner : nullptr), &p_path);
}

Ref<GLTFObjectModelProperty> GLTFDocument::import_object_model_property(const Ref<GLTFState> &p_state, const String &p_json_pointer) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("import_object_model_property")._native_ptr(), 1206708632);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFObjectModelProperty>()));
	return Ref<GLTFObjectModelProperty>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFObjectModelProperty>(_gde_method_bind, nullptr, (p_state != nullptr ? &p_state->_owner : nullptr), &p_json_pointer));
}

Ref<GLTFObjectModelProperty> GLTFDocument::export_object_model_property(const Ref<GLTFState> &p_state, const NodePath &p_node_path, Node *p_godot_node, int32_t p_gltf_node_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("export_object_model_property")._native_ptr(), 314209806);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<GLTFObjectModelProperty>()));
	int64_t p_gltf_node_index_encoded;
	PtrToArg<int64_t>::encode(p_gltf_node_index, &p_gltf_node_index_encoded);
	return Ref<GLTFObjectModelProperty>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<GLTFObjectModelProperty>(_gde_method_bind, nullptr, (p_state != nullptr ? &p_state->_owner : nullptr), &p_node_path, (p_godot_node != nullptr ? &p_godot_node->_owner : nullptr), &p_gltf_node_index_encoded));
}

void GLTFDocument::register_gltf_document_extension(const Ref<GLTFDocumentExtension> &p_extension, bool p_first_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("register_gltf_document_extension")._native_ptr(), 3752678331);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_first_priority_encoded;
	PtrToArg<bool>::encode(p_first_priority, &p_first_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, (p_extension != nullptr ? &p_extension->_owner : nullptr), &p_first_priority_encoded);
}

void GLTFDocument::unregister_gltf_document_extension(const Ref<GLTFDocumentExtension> &p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("unregister_gltf_document_extension")._native_ptr(), 2684415758);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, nullptr, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

PackedStringArray GLTFDocument::get_supported_gltf_extensions() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(GLTFDocument::get_class_static()._native_ptr(), StringName("get_supported_gltf_extensions")._native_ptr(), 2981934095);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr);
}

} // namespace godot
