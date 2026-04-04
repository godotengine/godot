/**************************************************************************/
/*  editor_export_plugin.cpp                                              */
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

#include <godot_cpp/classes/editor_export_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/editor_export_platform.hpp>
#include <godot_cpp/classes/editor_export_preset.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void EditorExportPlugin::add_shared_object(const String &p_path, const PackedStringArray &p_tags, const String &p_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_shared_object")._native_ptr(), 3098291045);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_tags, &p_target);
}

void EditorExportPlugin::add_file(const String &p_path, const PackedByteArray &p_file, bool p_remap) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_file")._native_ptr(), 527928637);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_remap_encoded;
	PtrToArg<bool>::encode(p_remap, &p_remap_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_file, &p_remap_encoded);
}

void EditorExportPlugin::add_apple_embedded_platform_project_static_lib(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_project_static_lib")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_apple_embedded_platform_framework(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_framework")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_apple_embedded_platform_embedded_framework(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_embedded_framework")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_apple_embedded_platform_plist_content(const String &p_plist_content) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_plist_content")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_plist_content);
}

void EditorExportPlugin::add_apple_embedded_platform_linker_flags(const String &p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_linker_flags")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

void EditorExportPlugin::add_apple_embedded_platform_bundle_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_bundle_file")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_apple_embedded_platform_cpp_code(const String &p_code) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_apple_embedded_platform_cpp_code")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_code);
}

void EditorExportPlugin::add_ios_project_static_lib(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_project_static_lib")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_ios_framework(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_framework")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_ios_embedded_framework(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_embedded_framework")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_ios_plist_content(const String &p_plist_content) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_plist_content")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_plist_content);
}

void EditorExportPlugin::add_ios_linker_flags(const String &p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_linker_flags")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

void EditorExportPlugin::add_ios_bundle_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_bundle_file")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::add_ios_cpp_code(const String &p_code) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_ios_cpp_code")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_code);
}

void EditorExportPlugin::add_macos_plugin_file(const String &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("add_macos_plugin_file")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

void EditorExportPlugin::skip() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("skip")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Variant EditorExportPlugin::get_option(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("get_option")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

Ref<EditorExportPreset> EditorExportPlugin::get_export_preset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("get_export_preset")._native_ptr(), 1610607222);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorExportPreset>()));
	return Ref<EditorExportPreset>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorExportPreset>(_gde_method_bind, _owner));
}

Ref<EditorExportPlatform> EditorExportPlugin::get_export_platform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorExportPlugin::get_class_static()._native_ptr(), StringName("get_export_platform")._native_ptr(), 282254641);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<EditorExportPlatform>()));
	return Ref<EditorExportPlatform>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<EditorExportPlatform>(_gde_method_bind, _owner));
}

void EditorExportPlugin::_export_file(const String &p_path, const String &p_type, const PackedStringArray &p_features) {}

void EditorExportPlugin::_export_begin(const PackedStringArray &p_features, bool p_is_debug, const String &p_path, uint32_t p_flags) {}

void EditorExportPlugin::_export_end() {}

bool EditorExportPlugin::_begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const PackedStringArray &p_features) const {
	return false;
}

Ref<Resource> EditorExportPlugin::_customize_resource(const Ref<Resource> &p_resource, const String &p_path) {
	return Ref<Resource>();
}

bool EditorExportPlugin::_begin_customize_scenes(const Ref<EditorExportPlatform> &p_platform, const PackedStringArray &p_features) const {
	return false;
}

Node *EditorExportPlugin::_customize_scene(Node *p_scene, const String &p_path) {
	return nullptr;
}

uint64_t EditorExportPlugin::_get_customization_configuration_hash() const {
	return 0;
}

void EditorExportPlugin::_end_customize_scenes() {}

void EditorExportPlugin::_end_customize_resources() {}

TypedArray<Dictionary> EditorExportPlugin::_get_export_options(const Ref<EditorExportPlatform> &p_platform) const {
	return TypedArray<Dictionary>();
}

Dictionary EditorExportPlugin::_get_export_options_overrides(const Ref<EditorExportPlatform> &p_platform) const {
	return Dictionary();
}

bool EditorExportPlugin::_should_update_export_options(const Ref<EditorExportPlatform> &p_platform) const {
	return false;
}

bool EditorExportPlugin::_get_export_option_visibility(const Ref<EditorExportPlatform> &p_platform, const String &p_option) const {
	return false;
}

String EditorExportPlugin::_get_export_option_warning(const Ref<EditorExportPlatform> &p_platform, const String &p_option) const {
	return String();
}

PackedStringArray EditorExportPlugin::_get_export_features(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return PackedStringArray();
}

String EditorExportPlugin::_get_name() const {
	return String();
}

bool EditorExportPlugin::_supports_platform(const Ref<EditorExportPlatform> &p_platform) const {
	return false;
}

PackedStringArray EditorExportPlugin::_get_android_dependencies(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return PackedStringArray();
}

PackedStringArray EditorExportPlugin::_get_android_dependencies_maven_repos(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return PackedStringArray();
}

PackedStringArray EditorExportPlugin::_get_android_libraries(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return PackedStringArray();
}

String EditorExportPlugin::_get_android_manifest_activity_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return String();
}

String EditorExportPlugin::_get_android_manifest_application_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return String();
}

String EditorExportPlugin::_get_android_manifest_element_contents(const Ref<EditorExportPlatform> &p_platform, bool p_debug) const {
	return String();
}

PackedByteArray EditorExportPlugin::_update_android_prebuilt_manifest(const Ref<EditorExportPlatform> &p_platform, const PackedByteArray &p_manifest_data) const {
	return PackedByteArray();
}

} // namespace godot
