/*************************************************************************/
/*  editor_export_plugin.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "editor_export_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "editor/editor_paths.h"
#include "editor/export/editor_export_platform.h"
#include "scene/resources/resource_format_text.h"

void EditorExportPlugin::set_export_preset(const Ref<EditorExportPreset> &p_preset) {
	if (p_preset.is_valid()) {
		export_preset = p_preset;
	}
}

Ref<EditorExportPreset> EditorExportPlugin::get_export_preset() const {
	return export_preset;
}

void EditorExportPlugin::add_file(const String &p_path, const Vector<uint8_t> &p_file, bool p_remap) {
	ExtraFile ef;
	ef.data = p_file;
	ef.path = p_path;
	ef.remap = p_remap;
	extra_files.push_back(ef);
}

void EditorExportPlugin::add_shared_object(const String &p_path, const Vector<String> &p_tags, const String &p_target) {
	shared_objects.push_back(SharedObject(p_path, p_tags, p_target));
}

void EditorExportPlugin::add_ios_framework(const String &p_path) {
	ios_frameworks.push_back(p_path);
}

void EditorExportPlugin::add_ios_embedded_framework(const String &p_path) {
	ios_embedded_frameworks.push_back(p_path);
}

Vector<String> EditorExportPlugin::get_ios_frameworks() const {
	return ios_frameworks;
}

Vector<String> EditorExportPlugin::get_ios_embedded_frameworks() const {
	return ios_embedded_frameworks;
}

void EditorExportPlugin::add_ios_plist_content(const String &p_plist_content) {
	ios_plist_content += p_plist_content + "\n";
}

String EditorExportPlugin::get_ios_plist_content() const {
	return ios_plist_content;
}

void EditorExportPlugin::add_ios_linker_flags(const String &p_flags) {
	if (ios_linker_flags.length() > 0) {
		ios_linker_flags += ' ';
	}
	ios_linker_flags += p_flags;
}

String EditorExportPlugin::get_ios_linker_flags() const {
	return ios_linker_flags;
}

void EditorExportPlugin::add_ios_bundle_file(const String &p_path) {
	ios_bundle_files.push_back(p_path);
}

Vector<String> EditorExportPlugin::get_ios_bundle_files() const {
	return ios_bundle_files;
}

void EditorExportPlugin::add_ios_cpp_code(const String &p_code) {
	ios_cpp_code += p_code;
}

String EditorExportPlugin::get_ios_cpp_code() const {
	return ios_cpp_code;
}

void EditorExportPlugin::add_macos_plugin_file(const String &p_path) {
	macos_plugin_files.push_back(p_path);
}

const Vector<String> &EditorExportPlugin::get_macos_plugin_files() const {
	return macos_plugin_files;
}

void EditorExportPlugin::add_ios_project_static_lib(const String &p_path) {
	ios_project_static_libs.push_back(p_path);
}

Vector<String> EditorExportPlugin::get_ios_project_static_libs() const {
	return ios_project_static_libs;
}

void EditorExportPlugin::_export_file_script(const String &p_path, const String &p_type, const Vector<String> &p_features) {
	GDVIRTUAL_CALL(_export_file, p_path, p_type, p_features);
}

void EditorExportPlugin::_export_begin_script(const Vector<String> &p_features, bool p_debug, const String &p_path, int p_flags) {
	GDVIRTUAL_CALL(_export_begin, p_features, p_debug, p_path, p_flags);
}

void EditorExportPlugin::_export_end_script() {
	GDVIRTUAL_CALL(_export_end);
}

// Customization

bool EditorExportPlugin::_begin_customize_resources(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) const {
	bool ret = false;
	if (GDVIRTUAL_CALL(_begin_customize_resources, p_platform, p_features, ret)) {
		return ret;
	}
	return false;
}

Ref<Resource> EditorExportPlugin::_customize_resource(const Ref<Resource> &p_resource, const String &p_path) {
	Ref<Resource> ret;
	if (GDVIRTUAL_REQUIRED_CALL(_customize_resource, p_resource, p_path, ret)) {
		return ret;
	}
	return Ref<Resource>();
}

bool EditorExportPlugin::_begin_customize_scenes(const Ref<EditorExportPlatform> &p_platform, const Vector<String> &p_features) const {
	bool ret = false;
	if (GDVIRTUAL_CALL(_begin_customize_scenes, p_platform, p_features, ret)) {
		return ret;
	}
	return false;
}

Node *EditorExportPlugin::_customize_scene(Node *p_root, const String &p_path) {
	Node *ret = nullptr;
	if (GDVIRTUAL_REQUIRED_CALL(_customize_scene, p_root, p_path, ret)) {
		return ret;
	}
	return nullptr;
}

uint64_t EditorExportPlugin::_get_customization_configuration_hash() const {
	uint64_t ret = 0;
	if (GDVIRTUAL_REQUIRED_CALL(_get_customization_configuration_hash, ret)) {
		return ret;
	}
	return 0;
}

void EditorExportPlugin::_end_customize_scenes() {
	GDVIRTUAL_CALL(_end_customize_scenes);
}

void EditorExportPlugin::_end_customize_resources() {
	GDVIRTUAL_CALL(_end_customize_resources);
}

String EditorExportPlugin::_get_name() const {
	String ret;
	if (GDVIRTUAL_REQUIRED_CALL(_get_name, ret)) {
		return ret;
	}
	return "";
}

void EditorExportPlugin::_export_file(const String &p_path, const String &p_type, const HashSet<String> &p_features) {
}

void EditorExportPlugin::_export_begin(const HashSet<String> &p_features, bool p_debug, const String &p_path, int p_flags) {
}

void EditorExportPlugin::skip() {
	skipped = true;
}

void EditorExportPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_shared_object", "path", "tags", "target"), &EditorExportPlugin::add_shared_object);
	ClassDB::bind_method(D_METHOD("add_ios_project_static_lib", "path"), &EditorExportPlugin::add_ios_project_static_lib);
	ClassDB::bind_method(D_METHOD("add_file", "path", "file", "remap"), &EditorExportPlugin::add_file);
	ClassDB::bind_method(D_METHOD("add_ios_framework", "path"), &EditorExportPlugin::add_ios_framework);
	ClassDB::bind_method(D_METHOD("add_ios_embedded_framework", "path"), &EditorExportPlugin::add_ios_embedded_framework);
	ClassDB::bind_method(D_METHOD("add_ios_plist_content", "plist_content"), &EditorExportPlugin::add_ios_plist_content);
	ClassDB::bind_method(D_METHOD("add_ios_linker_flags", "flags"), &EditorExportPlugin::add_ios_linker_flags);
	ClassDB::bind_method(D_METHOD("add_ios_bundle_file", "path"), &EditorExportPlugin::add_ios_bundle_file);
	ClassDB::bind_method(D_METHOD("add_ios_cpp_code", "code"), &EditorExportPlugin::add_ios_cpp_code);
	ClassDB::bind_method(D_METHOD("add_macos_plugin_file", "path"), &EditorExportPlugin::add_macos_plugin_file);
	ClassDB::bind_method(D_METHOD("skip"), &EditorExportPlugin::skip);

	GDVIRTUAL_BIND(_export_file, "path", "type", "features");
	GDVIRTUAL_BIND(_export_begin, "features", "is_debug", "path", "flags");
	GDVIRTUAL_BIND(_export_end);

	GDVIRTUAL_BIND(_begin_customize_resources, "platform", "features");
	GDVIRTUAL_BIND(_customize_resource, "resource", "path");

	GDVIRTUAL_BIND(_begin_customize_scenes, "platform", "features");
	GDVIRTUAL_BIND(_customize_scene, "scene", "path");

	GDVIRTUAL_BIND(_get_customization_configuration_hash);

	GDVIRTUAL_BIND(_end_customize_scenes);
	GDVIRTUAL_BIND(_end_customize_resources);

	GDVIRTUAL_BIND(_get_name);
}

EditorExportPlugin::EditorExportPlugin() {
}
