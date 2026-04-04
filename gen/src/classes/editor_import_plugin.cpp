/**************************************************************************/
/*  editor_import_plugin.cpp                                              */
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

#include <godot_cpp/classes/editor_import_plugin.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/string_name.hpp>

namespace godot {

Error EditorImportPlugin::append_import_external_resource(const String &p_path, const Dictionary &p_custom_options, const String &p_custom_importer, const Variant &p_generator_parameters) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorImportPlugin::get_class_static()._native_ptr(), StringName("append_import_external_resource")._native_ptr(), 320493106);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_path, &p_custom_options, &p_custom_importer, &p_generator_parameters);
}

String EditorImportPlugin::_get_importer_name() const {
	return String();
}

String EditorImportPlugin::_get_visible_name() const {
	return String();
}

int32_t EditorImportPlugin::_get_preset_count() const {
	return 0;
}

String EditorImportPlugin::_get_preset_name(int32_t p_preset_index) const {
	return String();
}

PackedStringArray EditorImportPlugin::_get_recognized_extensions() const {
	return PackedStringArray();
}

TypedArray<Dictionary> EditorImportPlugin::_get_import_options(const String &p_path, int32_t p_preset_index) const {
	return TypedArray<Dictionary>();
}

String EditorImportPlugin::_get_save_extension() const {
	return String();
}

String EditorImportPlugin::_get_resource_type() const {
	return String();
}

float EditorImportPlugin::_get_priority() const {
	return 0.0;
}

int32_t EditorImportPlugin::_get_import_order() const {
	return 0;
}

int32_t EditorImportPlugin::_get_format_version() const {
	return 0;
}

bool EditorImportPlugin::_get_option_visibility(const String &p_path, const StringName &p_option_name, const Dictionary &p_options) const {
	return false;
}

Error EditorImportPlugin::_import(const String &p_source_file, const String &p_save_path, const Dictionary &p_options, const TypedArray<String> &p_platform_variants, const TypedArray<String> &p_gen_files) const {
	return Error(0);
}

bool EditorImportPlugin::_can_import_threaded() const {
	return false;
}

} // namespace godot
