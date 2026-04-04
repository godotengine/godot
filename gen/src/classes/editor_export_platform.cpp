/**************************************************************************/
/*  editor_export_platform.cpp                                            */
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

#include <godot_cpp/classes/editor_export_platform.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

String EditorExportPlatform::get_os_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_os_name")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

Ref<EditorExportPreset> EditorExportPlatform::create_preset() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("create_preset")._native_ptr(), 2572397818);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorExportPreset>()));
	return Ref<EditorExportPreset>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorExportPreset>(_gde_method_bind, _owner));
}

Dictionary EditorExportPlatform::find_export_template(const String &p_template_file_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("find_export_template")._native_ptr(), 2248993622);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, &p_template_file_name);
}

Array EditorExportPlatform::get_current_presets() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_current_presets")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Array()));
	return ::godot::internal::_call_native_mb_ret<Array>(_gde_method_bind, _owner);
}

Dictionary EditorExportPlatform::save_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, bool p_embed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("save_pack")._native_ptr(), 3420080977);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	int8_t p_embed_encoded;
	PtrToArg<bool>::encode(p_embed, &p_embed_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_embed_encoded);
}

Dictionary EditorExportPlatform::save_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("save_zip")._native_ptr(), 1485052307);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path);
}

Dictionary EditorExportPlatform::save_pack_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("save_pack_patch")._native_ptr(), 1485052307);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path);
}

Dictionary EditorExportPlatform::save_zip_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("save_zip_patch")._native_ptr(), 1485052307);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path);
}

PackedStringArray EditorExportPlatform::gen_export_flags(BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("gen_export_flags")._native_ptr(), 2976483270);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_flags);
}

Error EditorExportPlatform::export_project_files(const Ref<EditorExportPreset> &p_preset, bool p_debug, const Callable &p_save_cb, const Callable &p_shared_cb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_project_files")._native_ptr(), 1063735070);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_save_cb, &p_shared_cb);
}

Error EditorExportPlatform::export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_project")._native_ptr(), 3879521245);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_flags);
}

Error EditorExportPlatform::export_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_pack")._native_ptr(), 3879521245);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_flags);
}

Error EditorExportPlatform::export_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_zip")._native_ptr(), 3879521245);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_flags);
}

Error EditorExportPlatform::export_pack_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_pack_patch")._native_ptr(), 608021658);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_patches, &p_flags);
}

Error EditorExportPlatform::export_zip_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("export_zip_patch")._native_ptr(), 608021658);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded, &p_path, &p_patches, &p_flags);
}

void EditorExportPlatform::clear_messages() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("clear_messages")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void EditorExportPlatform::add_message(EditorExportPlatform::ExportMessageType p_type, const String &p_category, const String &p_message) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("add_message")._native_ptr(), 782767225);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_category, &p_message);
}

int32_t EditorExportPlatform::get_message_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_message_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

EditorExportPlatform::ExportMessageType EditorExportPlatform::get_message_type(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_message_type")._native_ptr(), 2667287293);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorExportPlatform::ExportMessageType(0)));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return (EditorExportPlatform::ExportMessageType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_index_encoded);
}

String EditorExportPlatform::get_message_category(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_message_category")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

String EditorExportPlatform::get_message_text(int32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_message_text")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_index_encoded);
}

EditorExportPlatform::ExportMessageType EditorExportPlatform::get_worst_message_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_worst_message_type")._native_ptr(), 2580557466);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (EditorExportPlatform::ExportMessageType(0)));
	return (EditorExportPlatform::ExportMessageType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Error EditorExportPlatform::ssh_run_on_remote(const String &p_host, const String &p_port, const PackedStringArray &p_ssh_arg, const String &p_cmd_args, const Array &p_output, int32_t p_port_fwd) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("ssh_run_on_remote")._native_ptr(), 3163734797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_port_fwd_encoded;
	PtrToArg<int64_t>::encode(p_port_fwd, &p_port_fwd_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_host, &p_port, &p_ssh_arg, &p_cmd_args, &p_output, &p_port_fwd_encoded);
}

int64_t EditorExportPlatform::ssh_run_on_remote_no_wait(const String &p_host, const String &p_port, const PackedStringArray &p_ssh_args, const String &p_cmd_args, int32_t p_port_fwd) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("ssh_run_on_remote_no_wait")._native_ptr(), 3606362233);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_port_fwd_encoded;
	PtrToArg<int64_t>::encode(p_port_fwd, &p_port_fwd_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_host, &p_port, &p_ssh_args, &p_cmd_args, &p_port_fwd_encoded);
}

Error EditorExportPlatform::ssh_push_to_remote(const String &p_host, const String &p_port, const PackedStringArray &p_scp_args, const String &p_src_file, const String &p_dst_file) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("ssh_push_to_remote")._native_ptr(), 218756989);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_host, &p_port, &p_scp_args, &p_src_file, &p_dst_file);
}

Dictionary EditorExportPlatform::get_internal_export_files(const Ref<EditorExportPreset> &p_preset, bool p_debug) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_internal_export_files")._native_ptr(), 89550086);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Dictionary()));
	int8_t p_debug_encoded;
	PtrToArg<bool>::encode(p_debug, &p_debug_encoded);
	return ::godot::internal::_call_native_mb_ret<Dictionary>(_gde_method_bind, _owner, (p_preset != nullptr ? &p_preset->_owner : nullptr), &p_debug_encoded);
}

PackedStringArray EditorExportPlatform::get_forced_export_files(const Ref<EditorExportPreset> &p_preset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatform::get_class_static()._native_ptr(), StringName("get_forced_export_files")._native_ptr(), 1939331020);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, nullptr, (p_preset != nullptr ? &p_preset->_owner : nullptr));
}

} // namespace godot
