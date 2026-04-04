/**************************************************************************/
/*  resource_format_loader.cpp                                            */
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

#include <godot_cpp/classes/resource_format_loader.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

PackedStringArray ResourceFormatLoader::_get_recognized_extensions() const {
	return PackedStringArray();
}

bool ResourceFormatLoader::_recognize_path(const String &p_path, const StringName &p_type) const {
	return false;
}

bool ResourceFormatLoader::_handles_type(const StringName &p_type) const {
	return false;
}

String ResourceFormatLoader::_get_resource_type(const String &p_path) const {
	return String();
}

String ResourceFormatLoader::_get_resource_script_class(const String &p_path) const {
	return String();
}

int64_t ResourceFormatLoader::_get_resource_uid(const String &p_path) const {
	return 0;
}

PackedStringArray ResourceFormatLoader::_get_dependencies(const String &p_path, bool p_add_types) const {
	return PackedStringArray();
}

Error ResourceFormatLoader::_rename_dependencies(const String &p_path, const Dictionary &p_renames) const {
	return Error(0);
}

bool ResourceFormatLoader::_exists(const String &p_path) const {
	return false;
}

PackedStringArray ResourceFormatLoader::_get_classes_used(const String &p_path) const {
	return PackedStringArray();
}

Variant ResourceFormatLoader::_load(const String &p_path, const String &p_original_path, bool p_use_sub_threads, int32_t p_cache_mode) const {
	return Variant();
}

} // namespace godot
