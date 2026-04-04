/**************************************************************************/
/*  editor_export_platform_extension.cpp                                  */
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

#include <godot_cpp/classes/editor_export_platform_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/editor_export_preset.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void EditorExportPlatformExtension::set_config_error(const String &p_error_text) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatformExtension::get_class_static()._native_ptr(), StringName("set_config_error")._native_ptr(), 3089850668);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_error_text);
}

String EditorExportPlatformExtension::get_config_error() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatformExtension::get_class_static()._native_ptr(), StringName("get_config_error")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void EditorExportPlatformExtension::set_config_missing_templates(bool p_missing_templates) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatformExtension::get_class_static()._native_ptr(), StringName("set_config_missing_templates")._native_ptr(), 1695273946);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_missing_templates_encoded;
	PtrToArg<bool>::encode(p_missing_templates, &p_missing_templates_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_missing_templates_encoded);
}

bool EditorExportPlatformExtension::get_config_missing_templates() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlatformExtension::get_class_static()._native_ptr(), StringName("get_config_missing_templates")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

PackedStringArray EditorExportPlatformExtension::_get_preset_features(const Ref<EditorExportPreset> &p_preset) const {
	return PackedStringArray();
}

bool EditorExportPlatformExtension::_is_executable(const String &p_path) const {
	return false;
}

TypedArray<Dictionary> EditorExportPlatformExtension::_get_export_options() const {
	return TypedArray<Dictionary>();
}

bool EditorExportPlatformExtension::_should_update_export_options() {
	return false;
}

bool EditorExportPlatformExtension::_get_export_option_visibility(const Ref<EditorExportPreset> &p_preset, const String &p_option) const {
	return false;
}

String EditorExportPlatformExtension::_get_export_option_warning(const Ref<EditorExportPreset> &p_preset, const StringName &p_option) const {
	return String();
}

String EditorExportPlatformExtension::_get_os_name() const {
	return String();
}

String EditorExportPlatformExtension::_get_name() const {
	return String();
}

Ref<Texture2D> EditorExportPlatformExtension::_get_logo() const {
	return Ref<Texture2D>();
}

bool EditorExportPlatformExtension::_poll_export() {
	return false;
}

int32_t EditorExportPlatformExtension::_get_options_count() const {
	return 0;
}

String EditorExportPlatformExtension::_get_options_tooltip() const {
	return String();
}

Ref<Texture2D> EditorExportPlatformExtension::_get_option_icon(int32_t p_device) const {
	return Ref<Texture2D>();
}

String EditorExportPlatformExtension::_get_option_label(int32_t p_device) const {
	return String();
}

String EditorExportPlatformExtension::_get_option_tooltip(int32_t p_device) const {
	return String();
}

String EditorExportPlatformExtension::_get_device_architecture(int32_t p_device) const {
	return String();
}

void EditorExportPlatformExtension::_cleanup() {}

Error EditorExportPlatformExtension::_run(const Ref<EditorExportPreset> &p_preset, int32_t p_device, BitField<EditorExportPlatform::DebugFlags> p_debug_flags) {
	return Error(0);
}

Ref<Texture2D> EditorExportPlatformExtension::_get_run_icon() const {
	return Ref<Texture2D>();
}

bool EditorExportPlatformExtension::_can_export(const Ref<EditorExportPreset> &p_preset, bool p_debug) const {
	return false;
}

bool EditorExportPlatformExtension::_has_valid_export_configuration(const Ref<EditorExportPreset> &p_preset, bool p_debug) const {
	return false;
}

bool EditorExportPlatformExtension::_has_valid_project_configuration(const Ref<EditorExportPreset> &p_preset) const {
	return false;
}

PackedStringArray EditorExportPlatformExtension::_get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const {
	return PackedStringArray();
}

Error EditorExportPlatformExtension::_export_project(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return Error(0);
}

Error EditorExportPlatformExtension::_export_pack(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return Error(0);
}

Error EditorExportPlatformExtension::_export_zip(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return Error(0);
}

Error EditorExportPlatformExtension::_export_pack_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return Error(0);
}

Error EditorExportPlatformExtension::_export_zip_patch(const Ref<EditorExportPreset> &p_preset, bool p_debug, const String &p_path, const PackedStringArray &p_patches, BitField<EditorExportPlatform::DebugFlags> p_flags) {
	return Error(0);
}

PackedStringArray EditorExportPlatformExtension::_get_platform_features() const {
	return PackedStringArray();
}

String EditorExportPlatformExtension::_get_debug_protocol() const {
	return String();
}

void EditorExportPlatformExtension::_initialize() {}

} // namespace godot
