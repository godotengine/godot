/**************************************************************************/
/*  editor_scene_format_importer.cpp                                      */
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

#include <godot_cpp/classes/editor_scene_format_importer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>

namespace godot {

void EditorSceneFormatImporter::add_import_option(const String &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSceneFormatImporter::get_class_static()._native_ptr(), StringName("add_import_option")._native_ptr(), 402577236);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

void EditorSceneFormatImporter::add_import_option_advanced(Variant::Type p_type, const String &p_name, const Variant &p_default_value, PropertyHint p_hint, const String &p_hint_string, int32_t p_usage_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(EditorSceneFormatImporter::get_class_static()._native_ptr(), StringName("add_import_option_advanced")._native_ptr(), 3674075649);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_hint_encoded;
	PtrToArg<int64_t>::encode(p_hint, &p_hint_encoded);
	int64_t p_usage_flags_encoded;
	PtrToArg<int64_t>::encode(p_usage_flags, &p_usage_flags_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_type_encoded, &p_name, &p_default_value, &p_hint_encoded, &p_hint_string, &p_usage_flags_encoded);
}

PackedStringArray EditorSceneFormatImporter::_get_extensions() const {
	return PackedStringArray();
}

Object *EditorSceneFormatImporter::_import_scene(const String &p_path, uint32_t p_flags, const Dictionary &p_options) {
	return nullptr;
}

void EditorSceneFormatImporter::_get_import_options(const String &p_path) {}

Variant EditorSceneFormatImporter::_get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option) const {
	return Variant();
}

} // namespace godot
